#!/usr/bin/env python3
"""Turn SWORDS into concept-slot training data with HUMAN-VERIFIED positives and negatives.

Why
---
Every concept set this project has trained on so far came from WordNet or an LLM prompt, and none
of it has ever been validated. Measured pathologies in `data/*/youtube_clean/`: the observed
continuation is absent from ~94% of sets, 14% of sets contain a multi-word candidate, 11% contain
duplicates, and 59 candidates are literal LLM refusal strings. Any null result on that data is
uninterpretable -- it cannot distinguish "the objective does not work" from "the labels are noise".

SWORDS fixes exactly that. Each substitute carries 3 or 10 human TRUE/FALSE/UNSURE judgements, so:

  positives = substitutes humans accept in context          (acceptability >= --positive_threshold)
  negatives = substitutes humans reject in context          (acceptability <= --negative_threshold)

Those negatives are the thing `build_contrastive_dataset.py` was approximating with WordNet
co-hyponyms and wrong-sense mining at unknown validity. Here they are gold.

The official SWORDS *dev* split is deterministically divided into an 80/20 pilot train/validation
split. The official *test* split remains untouched until the objective and schedule are locked.
This provides a clean objective comparison on human-annotated data; it does not by itself identify
annotation quality as the cause of differences from the unmatched YouTube corpus.

Outputs (into --out_dir)
------------------------
  context_loss_{train,val,full}.csv          gold-inclusive [target] + positives
  context_loss_{train,val,full}_goldexcl.csv gold-exclusive positives only
  contrastive_{train,val,full}.csv           positives plus verified negatives
  vanilla_{train,val,full}.txt                matched observed-target replay
  context_syn_{train,val,full}.txt            Iyer et al. NCP data-augmentation corpus
  target_ids_{train,val}.txt                  benchmark evaluation allowlists
  slot_table.csv                  full provenance, including dropped rows
  prepare_report.json             every count and threshold used

Schema matches `data/*/youtube_clean_gold/` exactly, so `run_clm_sequence_ncp.py` and
`eval_concept_ppl_v3.py --gold_column gold_surface` consume it unchanged.

Usage
-----
  python prepare_swords_concept_sets.py \
      --swords_json data/swords/swords-v1.1_dev.json.gz \
      --out_dir data/swords/concept_dev
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Reuse the candidate hygiene rules already applied to the YouTube rebuild. This aligns one
# preprocessing choice; it does not make the otherwise unmatched corpora causally comparable.
MAX_CANDIDATE_CHARS = 40
MAX_CANDIDATE_WORDS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--swords_json", required=True)
    parser.add_argument("--test_json", default=None,
                        help="official test JSON, used only for a dev/test disjointness audit")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--positive_threshold", type=float, default=0.5,
                        help="SWORDS conservative threshold; >= this fraction of raters said TRUE")
    parser.add_argument("--negative_threshold", type=float, default=0.0,
                        help="<= this fraction said TRUE; 0.0 means unanimously rejected")
    parser.add_argument("--min_positives", type=int, default=2,
                        help="a slot needs at least this many alternatives besides the target")
    parser.add_argument("--max_negatives", type=int, default=10)
    parser.add_argument("--max_context_chars", type=int, default=600,
                        help="left-truncate long Enron/CoInCo threads, keeping the tail before the target")
    parser.add_argument("--pos", nargs="*", default=None,
                        help="restrict to these SWORDS POS tags, e.g. NOUN (default: keep all)")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    return parser.parse_args()


def normalize(candidate: str) -> Optional[str]:
    text = " ".join(str(candidate).split())
    if not text:
        return None
    if len(text) > MAX_CANDIDATE_CHARS or len(text.split()) > MAX_CANDIDATE_WORDS:
        return None
    return text


def load(path: str) -> Dict[str, Any]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def build_rows(data: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Counter]:
    by_target: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for substitute_id, substitute in data["substitutes"].items():
        labels = data["substitute_labels"].get(substitute_id) or []
        if not labels:
            continue
        acceptability = sum(label == "TRUE" for label in labels) / len(labels)
        by_target[substitute["target_id"]].append((substitute["substitute"], acceptability))

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train_fraction must lie strictly between 0 and 1")
    context_ids = sorted(data["contexts"])
    random.Random(args.split_seed).shuffle(context_ids)
    n_train = round(args.train_fraction * len(context_ids))
    train_context_ids = set(context_ids[:n_train])

    stats: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    for index, (target_id, target) in enumerate(sorted(data["targets"].items())):
        stats["targets_in"] += 1
        partition = "train" if target["context_id"] in train_context_ids else "val"
        stats[f"targets_{partition}_in"] += 1
        record: Dict[str, Any] = {
            "row_id": f"swords/{index:05d}", "target_id": target_id,
            "context_id": target["context_id"], "partition": partition,
            "pos": target.get("pos", "?"), "gold_surface": target["target"],
            "text": "", "positives": [], "negatives": [],
            "gold_inclusive": [], "keep": 0, "drop_reason": "",
        }

        if args.pos and target.get("pos") not in args.pos:
            record["drop_reason"] = "pos_filtered"; stats["drop_pos_filtered"] += 1
            rows.append(record); continue

        context = data["contexts"][target["context_id"]]["context"]
        offset, word = int(target["offset"]), target["target"]
        if context[offset : offset + len(word)] != word:
            record["drop_reason"] = "offset_mismatch"; stats["drop_offset_mismatch"] += 1
            rows.append(record); continue
        if normalize(word) is None:
            record["drop_reason"] = "target_unusable"; stats["drop_target_unusable"] += 1
            rows.append(record); continue

        # The causal prefix: everything left of the target, tail-truncated at a whitespace boundary.
        left = context[:offset]
        if len(left) > args.max_context_chars:
            left = left[-args.max_context_chars :]
            cut = left.find(" ")
            if cut > 0:
                left = left[cut + 1 :]
        left = " ".join(left.split())  # collapse the thread formatting; keeps tokenization sane
        if not left:
            record["drop_reason"] = "empty_prefix"; stats["drop_empty_prefix"] += 1
            rows.append(record); continue
        record["text"] = left

        gold_folded = word.casefold()
        positives, negatives, seen = [], [], {gold_folded}
        for candidate, acceptability in sorted(by_target.get(target_id, []), key=lambda p: -p[1]):
            clean = normalize(candidate)
            if clean is None:
                stats["candidate_dropped_unusable"] += 1
                continue
            key = clean.casefold()
            if key in seen:
                stats["candidate_dropped_duplicate"] += 1
                continue
            if acceptability >= args.positive_threshold:
                seen.add(key); positives.append(clean)
            elif acceptability <= args.negative_threshold and len(negatives) < args.max_negatives:
                seen.add(key); negatives.append(clean)

        if len(positives) < args.min_positives:
            record["drop_reason"] = "too_few_verified_positives"
            stats["drop_too_few_positives"] += 1
            rows.append(record); continue

        record.update(positives=positives, negatives=negatives,
                      gold_inclusive=[word] + positives, keep=1)
        stats["targets_kept"] += 1
        stats[f"targets_{partition}_kept"] += 1
        stats["positives_total"] += len(positives)
        stats["negatives_total"] += len(negatives)
        if not negatives:
            stats["kept_without_negatives"] += 1
        rows.append(record)
    return rows, stats


def write_csv(path: str, header: List[str], records: List[List[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(records)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_partition(out_dir: str, name: str, rows: List[Dict[str, Any]]) -> int:
    """Emit matched gold-inclusive, gold-exclusive, contrastive, replay and augmented views.

    Returns the number of augmentation lines written, which is the volume budget `N` that every
    Task-14 arm is matched to.

    The augmentation corpus reproduces Iyer et al. section 2.1.2 "NCP Data Augmentation": for a
    slot with n alternatives, emit n training instances, each targeting a different conceptually
    equivalent surface form.  It is built from the *gold-exclusive* positives, matching the
    convention in `data/syn/youtube_clean/context_syn_train.txt` -- 7,807 lines against 7,824
    gold-exclusive alternatives, of which only 86 coincide with a vanilla line.  The observed
    target is substituted out, not retained; `vanilla_{name}.txt` is where it lives.

    No LLM is involved: SWORDS positives are human acceptability judgements, so the substitution
    is mechanical.  The paper's Appendix-A prompting step applies only to its own corpora.
    """
    write_csv(os.path.join(out_dir, f"context_loss_{name}.csv"),
              ["text", "context_syn", "gold_surface", "row_id"],
              [[r["text"], repr(r["gold_inclusive"]), r["gold_surface"], r["row_id"]]
               for r in rows])
    write_csv(os.path.join(out_dir, f"context_loss_{name}_goldexcl.csv"),
              ["text", "context_syn", "gold_surface", "row_id"],
              [[r["text"], repr(r["positives"]), r["gold_surface"], r["row_id"]]
               for r in rows])
    write_csv(os.path.join(out_dir, f"contrastive_{name}.csv"),
              ["text", "positives", "negatives", "gold_surface", "row_id"],
              [[r["text"], repr(r["gold_inclusive"]), repr(r["negatives"]),
                r["gold_surface"], r["row_id"]] for r in rows])
    with open(os.path.join(out_dir, f"vanilla_{name}.txt"), "w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(f'{record["text"]} {record["gold_surface"]}\n')
    augmented = 0
    with open(os.path.join(out_dir, f"context_syn_{name}.txt"), "w", encoding="utf-8") as handle:
        for record in rows:
            for candidate in record["positives"]:
                handle.write(f'{record["text"]} {candidate}\n')
                augmented += 1
    return augmented


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load(args.swords_json)
    rows, stats = build_rows(data, args)
    kept = [r for r in rows if r["keep"]]
    if not kept:
        raise SystemExit("no slots survived; loosen --positive_threshold or --min_positives")

    train_rows = [r for r in kept if r["partition"] == "train"]
    val_rows = [r for r in kept if r["partition"] == "val"]
    augmentation_lines = {
        "train": write_partition(args.out_dir, "train", train_rows),
        "val": write_partition(args.out_dir, "val", val_rows),
        "full": write_partition(args.out_dir, "full", kept),
    }

    # Evaluation allowlists include every official dev target, not only rows that survived the
    # concept-training filter. This keeps held-out benchmark coverage independent of trainability.
    for partition in ("train", "val"):
        target_ids = sorted(r["target_id"] for r in rows if r["partition"] == partition)
        with open(os.path.join(args.out_dir, f"target_ids_{partition}.txt"),
                  "w", encoding="utf-8") as handle:
            handle.write("\n".join(target_ids) + "\n")

    write_csv(os.path.join(args.out_dir, "slot_table.csv"),
              ["row_id", "target_id", "context_id", "partition", "pos", "text",
               "gold_surface", "positives", "negatives",
               "gold_inclusive", "keep", "drop_reason"],
              [[r[k] for k in ["row_id", "target_id", "context_id", "partition", "pos",
                               "text", "gold_surface", "positives", "negatives",
                               "gold_inclusive", "keep", "drop_reason"]] for r in rows])

    train_target_ids = {r["target_id"] for r in rows if r["partition"] == "train"}
    val_target_ids = {r["target_id"] for r in rows if r["partition"] == "val"}
    train_context_ids = {r["context_id"] for r in rows if r["partition"] == "train"}
    val_context_ids = {r["context_id"] for r in rows if r["partition"] == "val"}
    assert train_target_ids.isdisjoint(val_target_ids)
    assert train_context_ids.isdisjoint(val_context_ids)

    test_audit = None
    if args.test_json:
        test = load(args.test_json)
        test_audit = {
            "target_id_overlap_with_dev": len(set(test["targets"]) & (train_target_ids | val_target_ids)),
            "context_id_overlap_with_dev": len(set(test["contexts"]) & (train_context_ids | val_context_ids)),
        }
        if any(test_audit.values()):
            raise RuntimeError(f"official SWORDS dev/test identifiers overlap: {test_audit}")

    report = {
        "source": args.swords_json,
        "source_sha256": sha256_file(args.swords_json),
        "test_source": args.test_json,
        "test_source_sha256": sha256_file(args.test_json) if args.test_json else None,
        "positive_threshold": args.positive_threshold,
        "negative_threshold": args.negative_threshold,
        "min_positives": args.min_positives,
        "max_negatives": args.max_negatives,
        "pos_filter": args.pos,
        "split_seed": args.split_seed,
        "train_fraction": args.train_fraction,
        "slots_kept": len(kept),
        "slots_train": len(train_rows),
        "slots_val": len(val_rows),
        # Volume budget for the matched Task-14 arms: one line per gold-exclusive positive.
        "augmentation_lines": augmentation_lines,
        "dev_test_disjointness": test_audit,
        "mean_positives": round(stats["positives_total"] / len(kept), 3),
        "mean_negatives": round(stats["negatives_total"] / len(kept), 3),
        "median_positives": statistics.median(len(r["positives"]) for r in kept),
        "pos_distribution": dict(Counter(r["pos"] for r in kept)),
        "counts": dict(sorted(stats.items())),
    }
    with open(os.path.join(args.out_dir, "prepare_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f'{stats["targets_in"]} targets in -> {len(kept)} slots kept '
          f'({len(kept) / max(stats["targets_in"], 1):.1%})')
    print(f'  leak-free pilot split: train={len(train_rows)} val={len(val_rows)} '
          f'(seed={args.split_seed}, grouped by official context_id)')
    print(f'  mean positives/slot {report["mean_positives"]}  '
          f'mean negatives/slot {report["mean_negatives"]}  '
          f'slots without negatives {stats["kept_without_negatives"]}')
    print(f'  POS: {report["pos_distribution"]}')
    print(f'  augmentation corpus: train={augmentation_lines["train"]} '
          f'val={augmentation_lines["val"]} full={augmentation_lines["full"]} lines')
    print(f'  drops: {{{", ".join(f"{k}={v}" for k, v in sorted(stats.items()) if k.startswith("drop_"))}}}')
    print(f'  wrote -> {args.out_dir}')


if __name__ == "__main__":
    main()
