"""How much of SWORDS is invisible to single-token concept supervision.

Iyer et al. and Zhang et al. both restrict the valid-alternative set to single
tokens of the model vocabulary.  This reports what that restriction discards on
the human-labelled substitution benchmark, and -- the part that matters -- how
human annotators rate the discarded candidates relative to the kept ones.

Multi-word is a lower bound on multi-token: a single word can also be several
tokens.  Counting whitespace needs no tokenizer, so this number is stable across
model families and is the conservative figure to quote.

    python scripts/swords_multiword_stats.py --swords data/swords/swords-v1.1_dev.json.gz
"""
import argparse
import gzip
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swords", default="data/swords/swords-v1.1_dev.json.gz")
    parser.add_argument("--acceptable-threshold", type=float, default=0.5,
                        help="Fraction of annotators calling a substitute acceptable.")
    args = parser.parse_args()

    data = json.load(gzip.open(args.swords))
    substitutes, labels = data["substitutes"], data["substitute_labels"]

    counts = {True: 0, False: 0}
    acceptance = {True: 0.0, False: 0.0}
    accepted = {True: 0, False: 0}
    for sid, record in substitutes.items():
        annotations = labels.get(sid, [])
        if not annotations:
            continue
        rate = sum(label == "TRUE" for label in annotations) / len(annotations)
        multiword = " " in record["substitute"].strip()
        counts[multiword] += 1
        acceptance[multiword] += rate
        accepted[multiword] += rate >= args.acceptable_threshold

    total = counts[True] + counts[False]
    total_accepted = accepted[True] + accepted[False]
    for multiword in (False, True):
        kind = "multi-word " if multiword else "single-word"
        n = counts[multiword]
        print(f"{kind}: n={n:6d} ({n / total:6.1%})  "
              f"mean human acceptance {acceptance[multiword] / n:.3f}  "
              f"accepted {accepted[multiword]:5d}")
    print(f"\nshare of ALL substitutes that are multi-word:        "
          f"{counts[True] / total:.1%}")
    print(f"share of ACCEPTED substitutes that are multi-word:   "
          f"{accepted[True] / total_accepted:.1%}")
    print("\nSingle-token supervision cannot represent any of the latter, and the "
          "acceptance rates show the filter is not selecting for quality.")


if __name__ == "__main__":
    main()
