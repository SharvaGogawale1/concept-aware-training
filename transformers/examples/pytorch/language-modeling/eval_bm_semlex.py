#!/usr/bin/env python3
"""bm-semlex minimal-pair evaluation: in-context synonym vs wrong-sense distractor.

What this is
------------
200 human-curated rows from SemCor + WordNet (github.com/ioana-ivan/bm-semlex), each
`target · synonym · distractor · token_index · sentence`.  In the given sentence the `synonym` is a
valid substitute for `target`; the `distractor` is another WordNet synonym of `target` drawn from a
*different sense*, and is not valid there.  Example:

    sentence : "Freed soil must be dispersed and protected against flocculation."
    target   : soil      synonym : dirt      distractor : territory

The task is a forced choice: does the model score the synonym above the distractor at that slot?
Chance is 0.50, so unlike a correlation this number is interpretable on its own.

Status in this project
----------------------
Supplementary, not a headline.  200 pairs from one unrefereed repo cannot carry a result, and the
distractor is by construction a wrong-sense item -- exactly the negative class
`build_contrastive_dataset.py` tried to mine from WordNet.  Its value is (a) a cheap sanity check
that a concept-trained model has not simply learned to spread mass over *any* WordNet neighbour,
which is the specific failure mode a set-marginal objective invites, and (b) prior art: the upstream
repo ran this style of test on OLMo-1B, OLMo-7B and Amber.

Two scoring modes, matching eval_swords.py:
  left  log p(candidate | prefix)                     -- what the concept objective optimizes
  full  candidate substituted in place, scored with its right context -- standard lexsub

Reports accuracy, a margin (mean log-prob difference), and a binomial 95% interval.

Usage
-----
  python eval_bm_semlex.py --checkpoints ckpt_a ckpt_b --tokenizer_path /path/to/Llama-3.2-1B \\
      --data data/bm_semlex/curated_200.tsv --results_json out/bm_semlex.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sequence_ncp_trainer import encode_candidate_continuation, sequence_log_probs_from_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--data", required=True, help="curated_200.tsv")
    parser.add_argument("--results_json", required=True)
    parser.add_argument("--modes", nargs="+", default=["left", "full"], choices=["left", "full"])
    parser.add_argument("--length_normalize", action="store_true", default=True)
    parser.add_argument("--no_length_normalize", dest="length_normalize", action="store_false")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def load_bm_semlex(path: str) -> List[Dict[str, Any]]:
    """Rows are `target, synonym, distractor, token_index, sentence` (tab separated, no header)."""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for line_no, fields in enumerate(csv.reader(handle, delimiter="\t")):
            if len(fields) < 5:
                continue
            target, synonym, distractor, index_text, sentence = fields[:5]
            tokens = sentence.split()
            try:
                index = int(index_text)
            except ValueError:
                continue
            # The published index is 0-based into whitespace tokens; verify rather than trust it,
            # and fall back to the first exact occurrence of the target.
            if not (0 <= index < len(tokens)) or tokens[index].strip(".,;:!?\"'").casefold() != target.casefold():
                matches = [
                    i for i, tok in enumerate(tokens)
                    if tok.strip(".,;:!?\"'").casefold() == target.casefold()
                ]
                if not matches:
                    continue
                index = matches[0]
            left = " ".join(tokens[:index])
            right = " ".join(tokens[index + 1 :])
            if not left:
                continue  # nothing to condition on; a causal LM cannot score this slot
            rows.append(
                {
                    "row_id": f"bm/{line_no:04d}",
                    "target": target, "synonym": synonym, "distractor": distractor,
                    "left": left, "right": (" " + right) if right else "",
                }
            )
    return rows


@torch.no_grad()
def _score(model: Any, tokenizer: Any, pairs: Sequence[Tuple[str, str]], block_size: int,
           batch_size: int, length_normalize: bool, device: torch.device) -> List[Optional[float]]:
    encoded, index_map = [], []
    for position, (context, continuation) in enumerate(pairs):
        item = encode_candidate_continuation(tokenizer, context, continuation, block_size)
        if item is not None:
            encoded.append(item)
            index_map.append(position)

    scores: List[Optional[float]] = [None] * len(pairs)
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        width = max(len(item["input_ids"]) for item in batch)

        def pad(key: str, value: int) -> torch.Tensor:
            return torch.tensor(
                [item[key] + [value] * (width - len(item[key])) for item in batch], device=device
            )

        logits = model(input_ids=pad("input_ids", tokenizer.pad_token_id),
                       attention_mask=pad("attention_mask", 0), use_cache=False).logits
        values = sequence_log_probs_from_logits(logits, pad("labels", -100)).float().cpu().tolist()
        for offset, value in enumerate(values):
            item = batch[offset]
            scores[index_map[start + offset]] = (
                value / max(len(item["continuation_ids"]), 1) if length_normalize else value
            )
    return scores


def wilson_interval(successes: int, total: int, z: float = 1.96) -> List[float]:
    """Wilson score interval -- correct near 0/1 where the normal approximation is not."""
    if not total:
        return [float("nan"), float("nan")]
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def evaluate_checkpoint(model: Any, tokenizer: Any, rows: Sequence[Dict[str, Any]],
                        args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for mode in args.modes:
        def build(field: str) -> List[Tuple[str, str]]:
            if mode == "left":
                return [(r["left"], r[field]) for r in rows]
            return [(r["left"], f'{r[field]}{r["right"]}') for r in rows]

        synonym = _score(model, tokenizer, build("synonym"), args.block_size,
                         args.batch_size, args.length_normalize, device)
        distractor = _score(model, tokenizer, build("distractor"), args.block_size,
                            args.batch_size, args.length_normalize, device)

        margins, per_row = [], []
        for row, s, d in zip(rows, synonym, distractor):
            if s is None or d is None:
                continue
            margins.append(s - d)
            per_row.append({"row_id": row["row_id"], "target": row["target"],
                            "margin": s - d, "correct": bool(s > d)})
        correct = sum(r["correct"] for r in per_row)
        result[mode] = {
            "n": len(per_row),
            "n_skipped": len(rows) - len(per_row),
            "accuracy": correct / len(per_row) if per_row else None,
            "accuracy_ci95": wilson_interval(correct, len(per_row)),
            "mean_margin": statistics.fmean(margins) if margins else None,
            "median_margin": statistics.median(margins) if margins else None,
            "per_row": per_row,
        }
    return result


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.float32 if device.type == "cpu"
             else {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[args.torch_dtype])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = load_bm_semlex(args.data)
    print(f"bm-semlex: {len(rows)} usable minimal pairs (chance accuracy = 0.500)")

    results = []
    for checkpoint in args.checkpoints:
        print(f"\n=== {checkpoint} ===")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
        model.config.pad_token_id = tokenizer.pad_token_id
        record = {"checkpoint": checkpoint, **evaluate_checkpoint(model, tokenizer, rows, args, device)}
        for mode in args.modes:
            summary = record[mode]
            low, high = summary["accuracy_ci95"]
            print(f"  {mode:5s} acc={summary['accuracy']:.4f} [{low:.3f}, {high:.3f}] "
                  f"margin={summary['mean_margin']:+.4f} (n={summary['n']})")
        results.append(record)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
    with open(args.results_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved {args.results_json}")


if __name__ == "__main__":
    main()
