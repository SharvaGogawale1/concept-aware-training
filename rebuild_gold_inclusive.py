#!/usr/bin/env python3
"""Build gold-inclusive concept splits from the clean YouTube splits.

Reads   data/{syn,hyp}/youtube_clean/
Writes  data/{syn,hyp}/youtube_clean_gold/     (originals are never modified)

Why
---
In `data/*/youtube_clean/context_loss_*.csv` the candidate set holds *alternatives to*
the observed continuation, with the observed continuation itself removed (it is present
in only ~3-8% of rows).  Both concept trainers encode a concept row as the context
prefix alone, so at the slot position the set-marginal term

    L_ncp = -log sum_{c in C} p(c | x)      ->    dL_ncp/dz_T = +alpha * p_T   when T not in C

is the *only* gradient acting on the gold logit: an unopposed downward push.  The
counteracting CLM signal lives on separate vanilla replay rows and reaches the slot only
through shared parameters.  Restoring T to the set flips that derivative's sign:

    T in C  ->  dL_ncp/dz_T = alpha * p_T * (1 - 1/S) <= 0     (S = sum_{c in C} p_c <= 1)

so the marginal objective can only redistribute mass *within* C, never off T.

Note this repair is specific to the set-marginal objective.  For the source paper's
mean-log form, L_mean = -(1/K) sum_c log p_c, the gold derivative is

    dL_mean/dz_T = p_T - 1/K

which still pushes the gold token down whenever p_T > 1/K.  Gold inclusion therefore
repairs `set_marginal` but does not remove the uniformization pressure in `paper_mean`.
Both arms are emitted here so that distinction can be measured rather than assumed.

Outputs (per relation directory)
--------------------------------
  slot_table.csv                  every input row, kept or dropped, with flags + provenance
  context_loss_{train,val}.csv    GOLD-INCLUSIVE candidates   (arm M-I)
  context_loss_{train,val}_goldexcl.csv
                                  GOLD-EXCLUSIVE, same rows, same quarantine  (arm M-X)
  quarantine.csv                  every removed candidate with a reason
  vanilla_{train,val}.txt         copied unchanged so the folder is self-contained

M-I and M-X are emitted over the *identical* kept-row set so the ablation is matched:
the only difference between the two files is the presence of the gold surface form.

Gold recovery rules
-------------------
  * same relation, same split only -- a train row is never repaired using val text
  * a row is recovered iff every source line having the row's `text` as a strict prefix
    agrees on the next surface form
  * boundary rule (declared, and the sole cause of +/-1 row disagreements between
    independent reimplementations):  gold = re.split(BOUNDARY, remainder.strip())[0]
    with BOUNDARY = r"[\s,.!?;:()\"]+"
  * <mask> rows, ambiguous rows and unmatched rows are dropped for this pilot

Usage
-----
  python rebuild_gold_inclusive.py                    # build
  python rebuild_gold_inclusive.py --dry_run          # report only, write nothing
  python rebuild_gold_inclusive.py --drop_slot_shift  # also drop slot-shift suspects
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

BOUNDARY = re.compile(r"[\s,.!?;:()\"]+")
SOURCE_BOUNDARY_CHARS = set(" \t\n'’“”\"“”,.!?;:()-")
MASK_TOKEN = "<mask>"

# Candidate-level quarantine thresholds (declared, not tuned).
MAX_CANDIDATE_CHARS = 40
MAX_CANDIDATE_WORDS = 4
ARTIFACT_MARKERS = (
    "cannot", "can not", "sorry", "as an ai", "i can help", "is there anything",
    "synonym", "hypernym", "import ", "from nltk", "wordnet", "here are", "note:",
    "the word", "in this context", "for the sentence", "```",
)

RELATIONS = ("syn", "hyp")
SPLITS = ("train", "val")


# --------------------------------------------------------------------------- io


def read_concept_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def parse_candidates(raw: Any) -> List[str]:
    try:
        value = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


# ------------------------------------------------------------------ quarantine


def classify_candidate(candidate: str) -> Tuple[str, str]:
    """Return (normalized_candidate, reason).  Empty reason means keep."""
    normalized = " ".join(str(candidate).split())
    if not normalized:
        return "", "empty_after_normalization"
    lowered = normalized.lower()
    if any(marker in lowered for marker in ARTIFACT_MARKERS):
        return normalized, "llm_artifact"
    if len(normalized) > MAX_CANDIDATE_CHARS or len(normalized.split()) > MAX_CANDIDATE_WORDS:
        return normalized, "overlong_phrase"
    return normalized, ""


def clean_candidate_set(candidates: Sequence[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Normalize, quarantine and case-insensitively deduplicate one candidate set."""
    kept: List[str] = []
    seen: set = set()
    removed: List[Tuple[str, str]] = []
    for candidate in candidates:
        normalized, reason = classify_candidate(candidate)
        if reason:
            removed.append((str(candidate), reason))
            continue
        key = normalized.casefold()
        if key in seen:
            removed.append((normalized, "duplicate_in_set"))
            continue
        seen.add(key)
        kept.append(normalized)
    return kept, removed


# --------------------------------------------------------------- gold recovery


def build_prefix_index(sentences: Sequence[str]) -> List[Tuple[str, int]]:
    """Deduplicated source lines with a stable id, longest first for cheap scanning."""
    unique = sorted(set(sentences))
    return [(sentence, index) for index, sentence in enumerate(unique)]


def recover_gold(context: str, index: Sequence[Tuple[str, int]]) -> Dict[str, Any]:
    """Recover the observed next surface form for `context` within one split."""
    matches = [
        (sentence, line_id) for sentence, line_id in index
        if sentence.startswith(context)
        and len(sentence) > len(context)
        and (context.endswith(tuple(SOURCE_BOUNDARY_CHARS))
             or sentence[len(context)] in SOURCE_BOUNDARY_CHARS)
    ]
    if not matches:
        return {"status": "no_source_match", "gold": None, "source_line_ids": []}

    golds: Dict[str, List[int]] = defaultdict(list)
    for sentence, line_id in matches:
        remainder = sentence[len(context):].strip()
        if not remainder:
            continue
        token = BOUNDARY.split(remainder)[0].strip()
        if token:
            golds[token].append(line_id)
    if not golds:
        return {"status": "empty_remainder", "gold": None, "source_line_ids": []}
    if len(golds) > 1:
        return {
            "status": "ambiguous_gold",
            "gold": None,
            "source_line_ids": sorted(i for ids in golds.values() for i in ids),
            "gold_candidates": sorted(golds),
        }
    gold, line_ids = next(iter(golds.items()))
    return {"status": "ok", "gold": gold, "source_line_ids": sorted(line_ids)}


# ------------------------------------------------------------------- rebuilding


def rebuild_split(
    relation: str,
    split: str,
    concept_csv: str,
    vanilla_txt: str,
    drop_slot_shift: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], Counter]:
    rows = read_concept_csv(concept_csv)
    index = build_prefix_index(read_lines(vanilla_txt))
    stats: Counter = Counter()
    slot_rows: List[Dict[str, Any]] = []
    quarantine: List[Dict[str, str]] = []

    for position, row in enumerate(rows):
        row_id = f"{relation}/{split}/{position:05d}"
        context = str(row.get("text", ""))
        raw_candidates = parse_candidates(row.get("context_syn", "[]"))
        stats["rows_in"] += 1

        record: Dict[str, Any] = {
            "row_id": row_id,
            "relation": relation,
            "split": split,
            "text": context,
            "original_candidates": raw_candidates,
            "alternatives": [],
            "gold_surface": "",
            "gold_inclusive_candidates": [],
            "source_line_ids": [],
            "recovery_status": "",
            "slot_shift_suspect": 0,
            "gold_already_present": 0,
            "n_quarantined": 0,
            "keep": 0,
            "drop_reason": "",
        }

        if MASK_TOKEN in context:
            record.update(recovery_status="mask_row", drop_reason="mask_row")
            stats["drop_mask"] += 1
            slot_rows.append(record)
            continue

        alternatives, removed = clean_candidate_set(raw_candidates)
        record["n_quarantined"] = len(removed)
        for candidate, reason in removed:
            quarantine.append(
                {"row_id": row_id, "relation": relation, "split": split,
                 "candidate": candidate, "reason": reason}
            )
            stats[f"quarantine_{reason}"] += 1

        recovery = recover_gold(context, index)
        record["recovery_status"] = recovery["status"]
        record["source_line_ids"] = recovery["source_line_ids"]

        if recovery["status"] != "ok":
            record["drop_reason"] = recovery["status"]
            stats[f"drop_{recovery['status']}"] += 1
            slot_rows.append(record)
            continue

        gold = recovery["gold"]
        record["gold_surface"] = gold
        stats["gold_recovered"] += 1

        # Slot-shift detector: the set describes a word the prefix has already passed.
        last_word = BOUNDARY.split(context.strip())[-1].casefold() if context.strip() else ""
        alt_keys = {alternative.casefold() for alternative in alternatives}
        if last_word and last_word in alt_keys:
            record["slot_shift_suspect"] = 1
            stats["slot_shift_suspect"] += 1
            if drop_slot_shift:
                record["drop_reason"] = "slot_shift_suspect"
                stats["drop_slot_shift_suspect"] += 1
                slot_rows.append(record)
                continue

        if gold.casefold() in alt_keys:
            record["gold_already_present"] = 1
            stats["gold_already_present"] += 1
            alternatives = [a for a in alternatives if a.casefold() != gold.casefold()]

        if not alternatives:
            record["drop_reason"] = "no_alternatives_after_quarantine"
            stats["drop_no_alternatives"] += 1
            slot_rows.append(record)
            continue

        record["alternatives"] = alternatives
        record["gold_inclusive_candidates"] = [gold] + alternatives
        record["keep"] = 1
        stats["rows_kept"] += 1
        stats["alternatives_total"] += len(alternatives)
        if len(alternatives) == 1:
            stats["kept_singleton_alternative"] += 1
        slot_rows.append(record)

    return slot_rows, quarantine, stats


# ----------------------------------------------------------------------- writers


def write_concept_csv(path: str, rows: Sequence[Dict[str, Any]], column: str) -> None:
    """Emit text / context_syn / gold_surface / row_id.

    `gold_surface` lets `eval_concept_ppl_v3.py --gold_column gold_surface` split the set
    marginal into gold and alternatives-only components; `row_id` keys the paired stats and
    is stable across the gold-inclusive and gold-exclusive variants of the same row.
    Trainers read only `text` and `context_syn`, so the extra columns are inert there.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "context_syn", "gold_surface", "row_id"])
        for row in rows:
            writer.writerow([row["text"], repr(row[column]), row["gold_surface"], row["row_id"]])


def write_slot_table(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "row_id", "relation", "split", "text", "gold_surface", "alternatives",
        "gold_inclusive_candidates", "original_candidates", "source_line_ids",
        "recovery_status", "slot_shift_suspect", "gold_already_present",
        "n_quarantined", "keep", "drop_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_quarantine(path: str, rows: Sequence[Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["row_id", "relation", "split", "candidate", "reason"]
        )
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------- validation


def assert_no_cross_split_repair(report: Dict[str, Any]) -> None:
    """Every kept row must cite source lines from its own split only.

    Enforced structurally: `recover_gold` is only ever handed the index built from the
    row's own split file.  This restates the invariant in the report so a future edit
    that breaks it is visible.
    """
    for relation in RELATIONS:
        for split in SPLITS:
            key = f"{relation}/{split}"
            assert report["splits"][key]["source_file"].endswith(f"vanilla_{split}.txt"), key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_suffix", default="youtube_clean_gold")
    parser.add_argument("--src_suffix", default="youtube_clean")
    parser.add_argument("--drop_slot_shift", action="store_true",
                        help="drop rows whose candidate set contains the prefix's last word")
    parser.add_argument("--dry_run", action="store_true", help="report only; write nothing")
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "boundary_regex": BOUNDARY.pattern,
        "max_candidate_chars": MAX_CANDIDATE_CHARS,
        "max_candidate_words": MAX_CANDIDATE_WORDS,
        "drop_slot_shift": bool(args.drop_slot_shift),
        "splits": {},
    }

    for relation in RELATIONS:
        src_dir = os.path.join(args.data_root, relation, args.src_suffix)
        out_dir = os.path.join(args.data_root, relation, args.out_suffix)
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)

        all_slot_rows: List[Dict[str, Any]] = []
        all_quarantine: List[Dict[str, str]] = []

        for split in SPLITS:
            concept_csv = os.path.join(src_dir, f"context_loss_{split}.csv")
            vanilla_txt = os.path.join(src_dir, f"vanilla_{split}.txt")
            slot_rows, quarantine, stats = rebuild_split(
                relation, split, concept_csv, vanilla_txt, args.drop_slot_shift
            )
            all_slot_rows.extend(slot_rows)
            all_quarantine.extend(quarantine)

            kept = [row for row in slot_rows if row["keep"]]
            report["splits"][f"{relation}/{split}"] = {
                "source_csv": concept_csv,
                "source_file": vanilla_txt,
                "rows_in": stats["rows_in"],
                "rows_kept": stats["rows_kept"],
                "keep_rate": round(stats["rows_kept"] / max(stats["rows_in"], 1), 4),
                "mean_alternatives_kept": round(
                    stats["alternatives_total"] / max(stats["rows_kept"], 1), 3
                ),
                "counts": dict(sorted(stats.items())),
            }

            if not args.dry_run:
                write_concept_csv(
                    os.path.join(out_dir, f"context_loss_{split}.csv"),
                    kept, "gold_inclusive_candidates",
                )
                write_concept_csv(
                    os.path.join(out_dir, f"context_loss_{split}_goldexcl.csv"),
                    kept, "alternatives",
                )
                shutil.copyfile(vanilla_txt, os.path.join(out_dir, f"vanilla_{split}.txt"))

            print(
                f"{relation}/{split:5s}  in={stats['rows_in']:5d}  kept={stats['rows_kept']:5d}"
                f"  ({stats['rows_kept'] / max(stats['rows_in'], 1):5.1%})"
                f"  gold_recovered={stats['gold_recovered']:5d}"
                f"  quarantined_cands={sum(v for k, v in stats.items() if k.startswith('quarantine_')):4d}"
                f"  slot_shift={stats['slot_shift_suspect']:3d}"
            )

        if not args.dry_run:
            write_slot_table(os.path.join(out_dir, "slot_table.csv"), all_slot_rows)
            write_quarantine(os.path.join(out_dir, "quarantine.csv"), all_quarantine)

    assert_no_cross_split_repair(report)

    report_path = os.path.join(args.data_root, "youtube_clean_gold_report.json")
    if not args.dry_run:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nReport: {report_path}")
    else:
        print("\n[dry run] nothing written")


if __name__ == "__main__":
    main()
