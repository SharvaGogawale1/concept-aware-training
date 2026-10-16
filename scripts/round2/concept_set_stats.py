"""Task 1.3: the |A_t| distribution per model, through the trainer's own preprocessing.

Loads each family's training file with train.ConceptDataset and flattens it with
ConceptLossTrainer._build_concept_targets, so |A_t| is exactly what the loss sees
(single-token synonyms, observed token and duplicates removed, first 10 kept).

    python scripts/round2/concept_set_stats.py [--split train]
Writes results/round2/concept_sets/A_distribution_<split>.json.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from types import SimpleNamespace

import r2common as C


def stats(tag, split):
    sys.path.insert(0, str(C.EXT))
    from transformers import AutoTokenizer
    from train import ConceptDataset, ConceptLossTrainer
    from conceptlib.config import TrainingConfig

    tokenizer = AutoTokenizer.from_pretrained(C.FAMILIES[tag])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    path = C.leaf(tag) / f"synonyms_{split}.jsonl"
    dataset = ConceptDataset(str(path), tokenizer, max_length=256)
    t = ConceptLossTrainer.__new__(ConceptLossTrainer)
    t.max_synonyms_per_content_word = TrainingConfig.max_synonyms_per_content_word
    t.max_content_words = None
    t.randomized_synonyms = False
    t.exclude_target = False
    t.state = SimpleNamespace(global_step=0)
    t.args = SimpleNamespace(seed=42)
    hist, pos_hist, sequences, tokens = Counter(), Counter(), 0, 0
    for item in dataset.data:
        sequences += 1
        tokens += int(item["attention_mask"].sum()) - 1
        flat = t._build_concept_targets([item["content_words"]], 1, len(item["input_ids"]) - 1,
                                        return_alternatives=True)
        for alts in flat[4]:
            hist[len(alts)] += 1
        for w in item["content_words"]:
            pos_hist[w.get("pos") or "?"] += 1
    n = sum(hist.values())
    return {
        "file": str(path), "file_sha256": C.sha256(path), "sequences": sequences,
        "supervised_positions": n, "predicted_tokens": tokens,
        "supervised_share_of_tokens": n / tokens if tokens else None,
        "A_hist": {str(k): hist[k] for k in sorted(hist)},
        "share_A_ge_2": sum(v for k, v in hist.items() if k >= 2) / n if n else None,
        "share_A_eq_0": hist[0] / n if n else None,
        "share_A_eq_1": hist[1] / n if n else None,
        "mean_A": sum(k * v for k, v in hist.items()) / n if n else None,
        "pos_hist": dict(pos_hist.most_common()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="train")
    ap.add_argument("--families", default=",".join(C.FAMILIES))
    args = ap.parse_args()
    out = {tag: stats(tag, args.split) for tag in args.families.split(",")}
    C.write_json(C.R2 / "concept_sets" / f"A_distribution_{args.split}.json", out)
    for tag, s in out.items():
        print(f"{tag:18s} positions={s['supervised_positions']:6d}  of tokens={s['supervised_share_of_tokens']:.3f}  "
              f"share|A|>=2={s['share_A_ge_2']:.3f}  "
              f"|A|=0 {s['share_A_eq_0']:.3f}  |A|=1 {s['share_A_eq_1']:.3f}  mean|A|={s['mean_A']:.2f}")
        print("   hist:", s["A_hist"])


if __name__ == "__main__":
    main()
