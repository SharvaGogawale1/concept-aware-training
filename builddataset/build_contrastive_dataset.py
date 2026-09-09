#!/usr/bin/env python3
"""
Task 3: Build a YouTube-only contrastive dataset with hard negatives.

Reads the existing context_loss CSV files (synonym and hypernym), then for each
context-prefix / positive-concept-set pair, uses WordNet to mine hard negatives:
  - Co-hyponyms: siblings of the INTENDED synset (same parent hypernym, different subtree)
  - Wrong-sense distractors: lemmas of the positives' non-intended synsets
    (July 2026 fix: the intended sense is now disambiguated via synonym
    intersection — see _intended_synsets — so this tier actually yields negatives)

Output:
    data/contrastive/youtube/contrastive_train.csv
    data/contrastive/youtube/contrastive_val.csv

Each output row has three columns:
    text       — context prefix (same as original)
    positives  — list of valid concept words (from original context_syn)
    negatives  — list of hard-negative words mined via WordNet

Also prints a coverage report: how many rows had at least one WordNet-derived negative.

Requirements:
    pip install nltk
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

import argparse
import ast
import os
import random
from collections import defaultdict

import pandas as pd

try:
    from nltk.corpus import wordnet as wn
    _WN_AVAILABLE = True
except ImportError:
    _WN_AVAILABLE = False
    print("WARNING: nltk not installed. Run: pip install nltk && python -c \"import nltk; nltk.download('wordnet')\"")


# ── WordNet helpers ──────────────────────────────────────────────────────────

def _synsets_for_word(word: str):
    """Return all WordNet synsets for a word (tries surface form and lowercased)."""
    word = word.strip()
    synsets = wn.synsets(word)
    if not synsets:
        synsets = wn.synsets(word.lower())
    if not synsets:
        # Try underscore form (WordNet stores multi-word lemmas with underscores)
        synsets = wn.synsets(word.replace(" ", "_"))
    return synsets


def _lemma_names_from_synset(synset) -> list:
    """All lemma name strings for a synset, cleaned."""
    return [l.name().replace("_", " ") for l in synset.lemmas()]


def _intended_synsets(positives: list) -> tuple:
    """
    Disambiguate the intended sense(s) of a positive concept set.

    July 2026 fix: the old code treated EVERY synset of every positive word as
    "the intended sense", which made the wrong-sense negative set empty by
    construction (0% mining coverage — the ablation silently degenerated to a
    no-negatives run).

    Key idea: the positives are (near-)synonyms, so they jointly disambiguate
    the sense — the synset(s) whose lemmas contain >= 2 distinct positives
    almost certainly denote the intended concept ({mom, mother, mommy} only
    intersect in the MOTHER synset). Fallback for singleton/non-intersecting
    sets: the most-frequent (first) sense of each word.

    Returns (intended_synsets: set, via_intersection: bool).
    """
    norm = {p.strip().lower().replace(" ", "_") for p in positives if p.strip()}
    intended = set()
    for word in positives:
        for ss in _synsets_for_word(word):
            lemmas = {l.name().lower() for l in ss.lemmas()}
            if len(lemmas & norm) >= 2:
                intended.add(ss)
    if intended:
        return intended, True
    for word in positives:
        synsets = _synsets_for_word(word)
        if synsets:
            intended.add(synsets[0])  # most-frequent sense
    return intended, False


def get_hard_negatives(
    positives: list,
    max_negatives: int = 10,
    pos_filter: bool = True,
    strategy: str = "all",
    stats: dict = None,
) -> list:
    """
    Mine hard negatives for a list of positive concept words.

    Strategy (run in priority order unless a specific strategy is selected):
    1. Co-hyponyms: siblings of the INTENDED sense(s) in the WordNet hierarchy.
    2. Wrong-sense distractors: lemmas of the positives' NON-intended synsets
       (July 2026 fix — previously empty by construction; see _intended_synsets).
       Chen's recommended safer option — co-hyponyms may still be valid in context.
    3. Same-POS fallback: random same-POS words from WordNet.

    Args:
        positives: list of positive concept words (strings)
        max_negatives: cap on returned negatives
        pos_filter: if True, only return negatives with the same POS as the
                    intended sense(s) (keeps distractors grammatically plausible)
        strategy: "all" | "co_hyponym" | "wrong_sense" | "same_pos" | "none"
                  "all" runs strategies 1-3 in priority order (default).
                  A named strategy runs ONLY that strategy — Task 8 ablations.
                  "none" returns [] — the no-negatives control (loss reduces to
                  CLM + positive NCP).
        stats: optional dict; increments "sense_via_intersection" /
               "sense_via_fallback" counters for the coverage report.
    """
    if not _WN_AVAILABLE or strategy == "none":
        return []

    positive_set = {p.strip().lower() for p in positives}
    positive_set.update({p.strip() for p in positives})

    intended_synsets, via_intersection = _intended_synsets(positives)
    if stats is not None:
        key = "sense_via_intersection" if via_intersection else "sense_via_fallback"
        stats[key] = stats.get(key, 0) + 1

    positive_pos = {ss.pos() for ss in intended_synsets}
    if not positive_pos:  # word not in WordNet at all
        for word in positives:
            for ss in _synsets_for_word(word):
                positive_pos.add(ss.pos())

    negatives = []
    seen = set(positive_set)

    # ── Strategy 1: co-hyponyms (siblings of the intended sense only) ────────
    if strategy in ("all", "co_hyponym"):
        for ss in intended_synsets:
            for hypernym in ss.hypernyms():
                for sibling in hypernym.hyponyms():
                    if sibling == ss:
                        continue
                    if pos_filter and sibling.pos() not in positive_pos:
                        continue
                    for lemma in sibling.lemmas():
                        word = lemma.name().replace("_", " ")
                        if word.lower() not in seen and word not in seen:
                            negatives.append(word)
                            seen.add(word.lower())
                            seen.add(word)
                            if len(negatives) >= max_negatives * 2:
                                break
                    if len(negatives) >= max_negatives * 2:
                        break
                if len(negatives) >= max_negatives * 2:
                    break

    # ── Strategy 2: wrong-sense distractors ─────────────────────────────────
    # Lemmas of the positives' NON-intended synsets. With the old definition
    # ("every synset of every positive is intended") this list was always
    # empty; intended_synsets now comes from synonym-intersection WSD.
    run_wrong_sense = strategy in ("all", "wrong_sense") and (
        strategy == "wrong_sense" or len(negatives) < max_negatives
    )
    if run_wrong_sense:
        def _hierarchy_related(ss):
            # A "wrong sense" that is an ancestor or descendant of an intended
            # synset is NOT a safe negative (e.g. mother.n.01 "female parent"
            # is the direct hypernym of ma.n.01 {mom, mommy, ...}).
            hypernym_closure = lambda s: s.hypernyms()
            for it in intended_synsets:
                if ss == it:
                    return True
                if it in ss.closure(hypernym_closure) or ss in it.closure(hypernym_closure):
                    return True
            return False

        for word in positives:
            all_synsets = _synsets_for_word(word)
            wrong_senses = [
                s for s in all_synsets
                if s not in intended_synsets and not _hierarchy_related(s)
            ]
            for ss in wrong_senses:
                if pos_filter and positive_pos and ss.pos() not in positive_pos:
                    continue
                for lemma in ss.lemmas():
                    cand = lemma.name().replace("_", " ")
                    if cand.lower() not in seen and cand not in seen:
                        negatives.append(cand)
                        seen.add(cand.lower())
                        seen.add(cand)
                if len(negatives) >= max_negatives * 2:
                    break

    # ── Strategy 3: same-POS fallback (random sample from WordNet) ──────────
    run_same_pos = strategy in ("all", "same_pos") and (
        strategy == "same_pos" or len(negatives) < max_negatives // 2
    )
    if run_same_pos and positive_pos:
        pos_tag = next(iter(positive_pos))
        all_synsets_pos = list(wn.all_synsets(pos=pos_tag))
        random.shuffle(all_synsets_pos)
        for ss in all_synsets_pos[:200]:
            for lemma in ss.lemmas():
                cand = lemma.name().replace("_", " ")
                if cand.lower() not in seen:
                    negatives.append(cand)
                    seen.add(cand.lower())
                    if len(negatives) >= max_negatives:
                        break
            if len(negatives) >= max_negatives:
                break

    random.shuffle(negatives)
    return negatives[:max_negatives]


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_contrastive_csv(
    input_csv: str,
    output_csv: str,
    max_negatives: int = 10,
    seed: int = 42,
    strategy: str = "all",
):
    """
    Build a contrastive CSV from a context_loss CSV.

    Args:
        input_csv: path to context_loss_train.csv or context_loss_val.csv
        output_csv: path where the contrastive CSV will be written
        max_negatives: max hard negatives per row
        seed: random seed for reproducibility
        strategy: negative mining strategy — "all" | "co_hyponym" | "wrong_sense" | "same_pos"
                  Pass a specific strategy for Task 8 ablation experiments.
    """
    random.seed(seed)

    df = pd.read_csv(input_csv)
    if "text" not in df.columns or "context_syn" not in df.columns:
        raise ValueError(f"Expected columns 'text' and 'context_syn' in {input_csv}")

    rows = []
    n_with_negatives = 0
    n_wordnet_miss = 0
    n_negatives_total = 0
    sense_stats = {}

    for _, row in df.iterrows():
        text = row["text"]
        try:
            positives = ast.literal_eval(str(row["context_syn"]))
        except Exception:
            positives = []

        if not isinstance(positives, list) or len(positives) == 0:
            continue

        positives = [str(p).strip().lstrip("\n") for p in positives if str(p).strip()]

        negatives = get_hard_negatives(
            positives, max_negatives=max_negatives, strategy=strategy, stats=sense_stats
        )

        if negatives:
            n_with_negatives += 1
            n_negatives_total += len(negatives)
        else:
            n_wordnet_miss += 1

        rows.append({
            "text": text,
            "positives": str(positives),
            "negatives": str(negatives),
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    # NOTE: "negative-mining coverage" (rows with >=1 mined negative) is a
    # different statistic from the "eval slot coverage" printed by the concept
    # eval scripts (rows with a scoreable concept). Report them separately.
    total = len(rows)
    coverage = n_with_negatives / total * 100 if total > 0 else 0
    avg_neg = n_negatives_total / n_with_negatives if n_with_negatives else 0.0
    print(f"  Strategy: {strategy}")
    print(f"  Wrote {total} rows to {output_csv}")
    print(f"  Negative-mining coverage: {n_with_negatives}/{total} rows have ≥1 hard negative ({coverage:.1f}%)")
    print(f"  Avg negatives per covered row: {avg_neg:.1f}")
    print(f"  Rows with no negatives mined: {n_wordnet_miss}")
    if strategy != "none" and sense_stats:
        print(f"  Sense disambiguation: {sense_stats.get('sense_via_intersection', 0)} via synonym "
              f"intersection, {sense_stats.get('sense_via_fallback', 0)} via first-sense fallback")
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Build contrastive dataset with hard negatives for YouTube domain"
    )
    parser.add_argument(
        "--syn_train", default="data/syn/youtube/context_loss_train.csv",
        help="Path to synonym context_loss_train.csv"
    )
    parser.add_argument(
        "--syn_val", default="data/syn/youtube/context_loss_val.csv",
        help="Path to synonym context_loss_val.csv"
    )
    parser.add_argument(
        "--hyp_train", default="data/hyp/youtube/context_loss_train.csv",
        help="Path to hypernym context_loss_train.csv"
    )
    parser.add_argument(
        "--hyp_val", default="data/hyp/youtube/context_loss_val.csv",
        help="Path to hypernym context_loss_val.csv"
    )
    parser.add_argument(
        "--output_dir", default="data/contrastive/youtube",
        help="Directory where output CSVs will be written"
    )
    parser.add_argument("--max_negatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source", choices=["syn", "hyp", "both"], default="both",
        help="Which concept type to use as source (synonym, hypernym, or both merged)"
    )
    parser.add_argument(
        "--strategy",
        choices=["all", "co_hyponym", "wrong_sense", "same_pos", "none"],
        default="all",
        help=(
            "Negative mining strategy. 'all' runs all three in priority order (default). "
            "Pass a specific strategy for Task 8 ablation experiments: "
            "'co_hyponym' (siblings of the intended sense), 'wrong_sense' (non-intended senses), "
            "'same_pos' (POS fallback), 'none' (no negatives — control run: loss reduces to "
            "CLM + positive NCP)."
        ),
    )
    args = parser.parse_args()

    if not _WN_AVAILABLE:
        print("ERROR: nltk wordnet not available. Install with:")
        print("  pip install nltk")
        print("  python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")
        return

    # Ensure WordNet data is downloaded
    try:
        wn.synsets("test")
    except Exception:
        import nltk
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    sources = []
    if args.source in ("syn", "both"):
        sources.append(("syn", args.syn_train, args.syn_val))
    if args.source in ("hyp", "both"):
        sources.append(("hyp", args.hyp_train, args.hyp_val))

    all_train_rows = []
    all_val_rows = []

    for source_name, train_path, val_path in sources:
        print(f"\nProcessing {source_name} train: {train_path}")
        train_df = build_contrastive_csv(
            train_path,
            os.path.join(args.output_dir, f"{source_name}_contrastive_train.csv"),
            args.max_negatives,
            args.seed,
            strategy=args.strategy,
        )
        all_train_rows.append(train_df)

        print(f"Processing {source_name} val:   {val_path}")
        val_df = build_contrastive_csv(
            val_path,
            os.path.join(args.output_dir, f"{source_name}_contrastive_val.csv"),
            args.max_negatives,
            args.seed,
            strategy=args.strategy,
        )
        all_val_rows.append(val_df)

    # Merged (combined syn+hyp) output
    if args.source == "both":
        merged_train = pd.concat(all_train_rows, ignore_index=True).drop_duplicates(subset=["text"])
        merged_val = pd.concat(all_val_rows, ignore_index=True).drop_duplicates(subset=["text"])
        merged_train_path = os.path.join(args.output_dir, "contrastive_train.csv")
        merged_val_path = os.path.join(args.output_dir, "contrastive_val.csv")
        merged_train.to_csv(merged_train_path, index=False)
        merged_val.to_csv(merged_val_path, index=False)
        print(f"\nMerged train: {len(merged_train)} rows → {merged_train_path}")
        print(f"Merged val:   {len(merged_val)} rows → {merged_val_path}")

    print("\nDone. To use in training, pass the merged CSV to run_clm_contrastive.py:")
    print(f"  --train_file {os.path.join(args.output_dir, 'contrastive_train.csv')}")
    print(f"  --validation_file {os.path.join(args.output_dir, 'contrastive_val.csv')}")


if __name__ == "__main__":
    main()
