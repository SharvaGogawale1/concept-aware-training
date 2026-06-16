#!/usr/bin/env python3
"""
Task 3: Build a YouTube-only contrastive dataset with hard negatives.

Reads the existing context_loss CSV files (synonym and hypernym), then for each
context-prefix / positive-concept-set pair, uses WordNet to mine hard negatives:
  - Co-hyponyms: siblings of a positive synset (same parent hypernym, different subtree)
  - Wrong-sense distractors: words sharing POS but from a different semantic field

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


def get_hard_negatives(
    positives: list,
    max_negatives: int = 10,
    pos_filter: bool = True,
) -> list:
    """
    Mine hard negatives for a list of positive concept words.

    Strategy (in priority order):
    1. Co-hyponyms: for each positive synset, find its hypernym, then collect
       lemmas from OTHER hyponyms of that hypernym (siblings in the hierarchy).
    2. Wrong-sense distractors: lemmas from synsets of a DIFFERENT sense of the
       same POS (e.g., "bank" as financial vs. river).
    3. Same-POS fallback: other words from WordNet with the same coarse POS.

    Args:
        positives: list of positive concept words (strings)
        max_negatives: cap on returned negatives
        pos_filter: if True, only return negatives with the same POS as at least
                    one positive (keeps distractors grammatically plausible)
    """
    if not _WN_AVAILABLE:
        return []

    positive_set = {p.strip().lower() for p in positives}
    positive_set.update({p.strip() for p in positives})

    # Collect POS tags from positives for filtering
    positive_pos = set()
    positive_synsets = set()
    for word in positives:
        for ss in _synsets_for_word(word):
            positive_pos.add(ss.pos())
            positive_synsets.add(ss)

    negatives = []
    seen = set(positive_set)

    # ── Strategy 1: co-hyponyms ──────────────────────────────────────────────
    for ss in positive_synsets:
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
    if len(negatives) < max_negatives:
        for word in positives:
            all_synsets = _synsets_for_word(word)
            # Only consider senses NOT in the positive synset group
            non_positive_senses = [s for s in all_synsets if s not in positive_synsets]
            for ss in non_positive_senses:
                for lemma in ss.lemmas():
                    cand = lemma.name().replace("_", " ")
                    if cand.lower() not in seen and cand not in seen:
                        negatives.append(cand)
                        seen.add(cand.lower())
                        seen.add(cand)
                if len(negatives) >= max_negatives * 2:
                    break

    # ── Strategy 3: same-POS fallback (random sample from WordNet) ──────────
    if len(negatives) < max_negatives // 2 and positive_pos:
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

    # Shuffle and cap
    random.shuffle(negatives)
    return negatives[:max_negatives]


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_contrastive_csv(input_csv: str, output_csv: str, max_negatives: int = 10, seed: int = 42):
    """
    Build a contrastive CSV from a context_loss CSV.

    Args:
        input_csv: path to context_loss_train.csv or context_loss_val.csv
        output_csv: path where the contrastive CSV will be written
        max_negatives: max hard negatives per row
        seed: random seed for reproducibility
    """
    random.seed(seed)

    df = pd.read_csv(input_csv)
    if "text" not in df.columns or "context_syn" not in df.columns:
        raise ValueError(f"Expected columns 'text' and 'context_syn' in {input_csv}")

    rows = []
    n_with_negatives = 0
    n_wordnet_miss = 0

    for _, row in df.iterrows():
        text = row["text"]
        try:
            positives = ast.literal_eval(str(row["context_syn"]))
        except Exception:
            positives = []

        if not isinstance(positives, list) or len(positives) == 0:
            continue

        # Clean up positives (strip whitespace, newline prefixes)
        positives = [str(p).strip().lstrip("\n") for p in positives if str(p).strip()]

        negatives = get_hard_negatives(positives, max_negatives=max_negatives)

        if negatives:
            n_with_negatives += 1
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

    total = len(rows)
    coverage = n_with_negatives / total * 100 if total > 0 else 0
    print(f"  Wrote {total} rows to {output_csv}")
    print(f"  WordNet coverage: {n_with_negatives}/{total} rows have ≥1 hard negative ({coverage:.1f}%)")
    print(f"  WordNet misses (no negatives found): {n_wordnet_miss}")
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
        )
        all_train_rows.append(train_df)

        print(f"Processing {source_name} val:   {val_path}")
        val_df = build_contrastive_csv(
            val_path,
            os.path.join(args.output_dir, f"{source_name}_contrastive_val.csv"),
            args.max_negatives,
            args.seed,
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
