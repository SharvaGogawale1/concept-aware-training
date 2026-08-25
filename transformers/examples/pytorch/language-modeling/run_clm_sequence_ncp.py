#!/usr/bin/env python3
"""Train exact sequence-level NCP objectives on per-example candidate sets.

Intentionally separate from the historical NCP runners, and the canonical runner for the
Tasks-13 audit.  Writing ``s_c = sum_t log p(c_t | x, c_<t)`` for the full-sequence log
probability of candidate ``c`` over a set ``C`` of size ``n``:

* ``none``                 no concept term -- the matched CLM / alpha=0 control
* ``paper_mean``           ``-(1/n) sum_c s_c``            Iyer et al.'s NCP loss
* ``set_marginal``         ``-log sum_c exp(s_c)``         the full-sequence set marginal
* ``set_marginal_mean``    ``-log((1/n) sum_c exp(s_c))``  = set_marginal + log n
* ``set_marginal_scaled``  ``-(1/n) log sum_c exp(s_c)``   = set_marginal / n

``set_marginal_mean`` shifts by ``log n``, which is constant in the parameters, so it is
gradient-identical to ``set_marginal`` and trains to the same model; it is useful as a metric
(it removes the set-size bias) rather than as a training arm.  ``set_marginal_scaled`` divides
instead, which does change the gradient.

Total loss is ``base_loss_weight * CLM + ncp_alpha * concept + contrast_beta * InfoNCE``.  Set
``--base_loss_weight 0`` to reproduce the paper's equation, which carries no CLM term.

``--train_file`` accepts a concept CSV (``text``, candidate column, ``gold_surface``) or a plain
``.txt`` of one sentence per line, which is treated as CLM-only rows.  Routing the plain-text arms
through this script keeps their schedule, logging and loss curves comparable with the concept arms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from collections import Counter
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

from sequence_ncp_trainer import (
    OBJECTIVES,
    SequenceNCPDataCollator,
    SequenceNCPTrainer,
    parse_candidate_list,
    tokenize_concept_record,
)


LOGGER = logging.getLogger("run_clm_sequence_ncp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", required=True)
    parser.add_argument("--replay_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), required=True)
    parser.add_argument("--ncp_alpha", type=float, default=0.5)
    parser.add_argument(
        "--base_loss_weight", type=float, default=1.0,
        help="weight on ordinary CLM/gold-continuation NLL (paper equation: 0; hybrids: 1)",
    )
    parser.add_argument("--required_coverage", type=float, default=0.99)
    parser.add_argument("--candidate_column", default="context_syn")
    parser.add_argument("--gold_column", default="gold_surface",
                        help="observed continuation used for matched gold-slot NTP")
    parser.add_argument("--negative_column", default="negatives",
                        help="CSV column of hard negatives; used only when --contrast_beta > 0")
    parser.add_argument("--contrast_beta", type=float, default=0.0,
                        help="weight on the InfoNCE term over mined negatives (0 disables it)")
    parser.add_argument("--candidate_microbatch_size", type=int, default=0,
                        help="candidate sequences per differentiable forward (0 = all)")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--save_strategy", choices=["no", "epoch"], default="no",
                        help="Task-14 defaults to no intermediate checkpoints; one final model is saved")
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite_cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite_output_dir", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--do_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report_to", default="none")
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        help="HF optimizer name. Use adamw_bnb_8bit for 3B-class models: fp32 Adam states are\n             ~4x the parameter count and are what pushes a 3B full fine-tune past 40GB.",
    )
    parser.add_argument(
        "--save_only_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write weights only -- no optimizer/scheduler/RNG state. Keeps ephemeral scratch\n             directories small; the state files are never needed once eval has run.",
    )
    parser.add_argument("--preprocess_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preprocessing_report", default=None)
    parser.add_argument("--preprocessing_cache_dir", default=None)
    parser.add_argument(
        "--deduplicate_text_rows", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "deduplicate plain-text CLM/replay rows; disable for Iyer et al.'s repeated-original "
            "NTP baseline and matched-volume Task-13 arms"
        ),
    )
    parser.add_argument(
        "--forbidden_output_root", action="append", default=[],
        help="hard-fail when output_dir is inside this root; repeatable (for example /content/drive)",
    )
    return parser.parse_args()


def _read_concept_csv(path: str, candidate_column: str, split: str,
                      negative_column: str = "negatives",
                      gold_column: str = "gold_surface") -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a text column")
        if candidate_column not in reader.fieldnames:
            for fallback in ("context_syn", "positives", "dict_syn"):
                if fallback in reader.fieldnames:
                    candidate_column = fallback
                    break
            else:
                raise ValueError(f"{path} has no candidate column")
        has_negatives = negative_column in (reader.fieldnames or [])
        if gold_column not in (reader.fieldnames or []):
            raise ValueError(
                f"{path} has no {gold_column!r} column; concept rows require explicit gold-slot NTP"
            )
        rows = []
        for index, row in enumerate(reader):
            candidates = parse_candidate_list(row[candidate_column])
            gold_surface = str(row.get(gold_column, "") or "").strip()
            if candidates and not gold_surface:
                raise ValueError(f"{path} row {index} has candidates but no {gold_column}")
            rows.append(
                {
                    "row_id": str(row.get("row_id") or f"{split}:concept:{index}"),
                    "text": str(row["text"]),
                    "context_syn": candidates,
                    "negatives": parse_candidate_list(row[negative_column]) if has_negatives else [],
                    "gold_surface": gold_surface,
                }
            )
    return rows


def _read_replay(
    path: str, split: str = "train", deduplicate: bool = True
) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    selected_lines = list(dict.fromkeys(lines)) if deduplicate else lines
    return [
        {"row_id": f"{split}:replay:{index}", "text": text, "context_syn": []}
        for index, text in enumerate(selected_lines)
    ]


def _read_train_rows(path: str, candidate_column: str, split: str,
                     negative_column: str = "negatives",
                     gold_column: str = "gold_surface",
                     deduplicate_text_rows: bool = True) -> List[Dict[str, Any]]:
    """Dispatch on file type.

    A ``.txt`` file is one sentence per line with no concept supervision -- the pure-CLM arms
    (A0 vanilla, A1 data augmentation).  Routing them through this script rather than
    ``run_clm.py`` is what makes their optimizer schedule, logging and loss curves directly
    comparable with the concept arms; the Plan-A round could not compare them because the two
    scripts ran different schedules.
    """
    if path.lower().endswith((".txt", ".text")):
        return _read_replay(path, split, deduplicate=deduplicate_text_rows)
    return _read_concept_csv(path, candidate_column, split, negative_column, gold_column)


def _prepare_split(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    block_size: int,
    overwrite_cache: bool,
    max_samples: Optional[int],
    split: str,
    cache_dir: Optional[str] = None,
) -> Tuple[Dataset, Dict[str, Any]]:
    if max_samples is not None:
        rows = rows[:max_samples]
    rows_sha256 = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    raw = Dataset.from_list(rows)
    encode = partial(
        tokenize_concept_record,
        tokenizer=tokenizer,
        block_size=block_size,
        candidate_column="context_syn",
        negative_column="negatives",
        gold_column="gold_surface",
        require_gold_for_concept=True,
    )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_material = backend.to_str() if backend is not None else json.dumps(tokenizer.get_vocab(), sort_keys=True)
    tokenizer_sha256 = hashlib.sha256(tokenizer_material.encode("utf-8")).hexdigest()
    cache_key = f"b{block_size}_{tokenizer_sha256[:8]}_{rows_sha256[:8]}"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    map_cache = os.path.join(cache_dir, f"{split}_{cache_key}_encoded.arrow") if cache_dir else None
    filter_cache = os.path.join(cache_dir, f"{split}_{cache_key}_kept.arrow") if cache_dir else None
    mapped = raw.map(
        encode,
        batched=False,
        load_from_cache_file=not overwrite_cache,
        cache_file_name=map_cache,
        desc=f"Encoding exact candidates [{split}]",
    )
    reasons = Counter(mapped["drop_reason"])
    kept = mapped.filter(
        lambda example: bool(example["keep"]),
        load_from_cache_file=not overwrite_cache,
        cache_file_name=filter_cache,
    )
    eligible_raw = sum(bool(parse_candidate_list(row.get("context_syn", []))) for row in rows)
    eligible_kept = sum(int(value) for value in kept["eligible"])
    supervised_kept = sum(bool(value) for value in kept["candidate_input_ids"])
    digest_payload = [
        {
            "row_id": row_id,
            "eligible": int(eligible),
            "candidate_token_ids": candidate_ids,
            "candidate_labels": candidate_labels,
        }
        for row_id, eligible, candidate_ids, candidate_labels in zip(
            kept["row_id"], kept["eligible"], kept["candidate_token_ids"], kept["candidate_labels"]
        )
    ]
    metadata_sha256 = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    stats = {
        "split": split,
        "rows_raw": len(rows),
        "rows_kept": len(kept),
        "base_supervised_tokens": sum(
            token != -100 for labels in kept["labels"] for token in labels
        ),
        "candidate_sequences": sum(len(group) for group in kept["candidate_labels"]),
        "candidate_supervised_tokens": sum(
            token != -100
            for group in kept["candidate_labels"]
            for labels in group
            for token in labels
        ),
        "negative_sequences": sum(len(group) for group in kept["negative_labels"]),
        "negative_supervised_tokens": sum(
            token != -100
            for group in kept["negative_labels"]
            for labels in group
            for token in labels
        ),
        "eligible_raw": eligible_raw,
        "eligible_after_declared_exclusions": eligible_kept,
        "supervised_after_declared_exclusions": supervised_kept,
        "supervision_coverage": supervised_kept / max(eligible_kept, 1),
        "drop_reasons": {key: value for key, value in reasons.items() if key},
        "left_truncated_rows": sum(int(value) > 0 for value in kept["n_left_truncated"]),
        "duplicate_context_rows_preserved": len(rows) - len({row["text"] for row in rows}),
        "candidate_metadata_sha256": metadata_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "raw_rows_sha256": rows_sha256,
    }
    if eligible_kept and stats["supervision_coverage"] < 0.99:
        raise RuntimeError(f"preprocessing concept coverage below 99%: {stats}")
    return kept, stats


def _dtype(name: str):
    if name == "auto":
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)

    output_dir = os.path.abspath(args.output_dir)
    for root in args.forbidden_output_root:
        forbidden = os.path.abspath(root)
        if os.path.commonpath([output_dir, forbidden]) == forbidden:
            raise ValueError(f"output_dir must be ephemeral; {output_dir} is inside {forbidden}")

    tokenizer_path = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("canonical tokenizer must define eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_rows = _read_train_rows(args.train_file, args.candidate_column, "train",
                                  args.negative_column, args.gold_column,
                                  args.deduplicate_text_rows)
    if args.replay_file:
        train_rows.extend(
            _read_replay(args.replay_file, deduplicate=args.deduplicate_text_rows)
        )
    validation_rows = _read_train_rows(args.validation_file, args.candidate_column, "validation",
                                       args.negative_column, args.gold_column,
                                       args.deduplicate_text_rows)

    train_dataset, train_stats = _prepare_split(
        train_rows,
        tokenizer,
        args.block_size,
        args.overwrite_cache,
        args.max_train_samples,
        "train",
        args.preprocessing_cache_dir,
    )
    eval_dataset, eval_stats = _prepare_split(
        validation_rows,
        tokenizer,
        args.block_size,
        args.overwrite_cache,
        args.max_eval_samples,
        "validation",
        args.preprocessing_cache_dir,
    )
    LOGGER.info("train preprocessing: %s", json.dumps(train_stats, sort_keys=True))
    LOGGER.info("validation preprocessing: %s", json.dumps(eval_stats, sort_keys=True))

    if args.preprocess_only:
        report = {"train_preprocessing": train_stats, "validation_preprocessing": eval_stats}
        report_path = args.preprocessing_report or os.path.join(args.output_dir, "preprocessing_report.json")
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=_dtype(args.torch_dtype))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        eval_strategy="epoch" if args.do_eval else "no",
        save_strategy=args.save_strategy if args.do_train else "no",
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        optim=args.optim,
        load_best_model_at_end=bool(
            args.load_best_model_at_end and args.do_train and args.do_eval
            and args.save_strategy != "no"
        ),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SequenceNCPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=SequenceNCPDataCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        objective=args.objective,
        alpha=args.ncp_alpha,
        base_loss_weight=args.base_loss_weight,
        contrast_beta=args.contrast_beta,
        required_coverage=args.required_coverage,
        candidate_microbatch_size=args.candidate_microbatch_size,
    )

    run_summary: Dict[str, Any] = {
        "objective": args.objective,
        "ncp_alpha": args.ncp_alpha,
        "base_loss_weight": args.base_loss_weight,
        "contrast_beta": args.contrast_beta,
        "candidate_microbatch_size": args.candidate_microbatch_size,
        "save_strategy": args.save_strategy,
        "deduplicate_text_rows": args.deduplicate_text_rows,
        "seed": args.seed,
        "model_name_or_path": args.model_name_or_path,
        "train_file": args.train_file,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "train_preprocessing": train_stats,
        "validation_preprocessing": eval_stats,
    }
    if args.do_train:
        train_result = trainer.train()
        # A pure replay/CLM run legitimately has no concept rows.  Concept-bearing arms must
        # still hard-fail below the requested coverage threshold.
        if train_stats["eligible_after_declared_exclusions"]:
            coverage = trainer.assert_training_coverage()
        else:
            coverage = trainer.concept_coverage_stats()
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        trainer.save_metrics("train", train_result.metrics)
        if not args.save_only_model:
            trainer.save_state()
        run_summary["training_concept_coverage"] = coverage
        run_summary["train_metrics"] = train_result.metrics
        # Per-logging-step history: total loss plus the clm / concept / contrast components that
        # SequenceNCPTrainer.log() injects.  Kept in the summary JSON because --save_only_model
        # suppresses trainer_state.json, and because the checkpoint directory is deleted after
        # evaluation -- this is the only surviving record of how the loss actually moved.
        run_summary["log_history"] = trainer.state.log_history
    if args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)
        run_summary["eval_metrics"] = eval_metrics

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "sequence_ncp_run.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, sort_keys=True)
    print(json.dumps(run_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
