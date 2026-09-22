#!/usr/bin/env python3
"""Go/no-go for graded concept supervision: does the extraction's own confidence rank
human-accepted substitutes above rejected ones?

Every objective so far pushes the within-set distribution toward some target w:
Zhang toward the model's own q, the uniform arms toward u.  Neither carries any
information about WHICH alternative is valid, so neither can produce a large SWORDS
gain.  The next member of the family is w taken from the extraction's confidence --
Zhang's Method 1 scores candidates by input-embedding cosine to the target.  That is
only worth training if the cosine itself separates accepted from rejected substitutes
in SWORDS.  This measures exactly that, within target, the way GAP is computed.

Chance is AUROC .50.  For reference, the PRETRAINED model's own within-target AUROC on
SWORDS dev (left mode) is .66 for Llama-3.2-1B and .66 for Qwen3-1.7B-Base.  If cosine
sits near .50 there is no self-supervised w to learn from and the experiment is dead
before it costs a GPU-hour.

    python scripts/diag_confidence_vs_human.py --model Qwen/Qwen3-1.7B-Base \
        --swords data/swords/swords-v1.1_dev.json.gz --out outputs/diag_cosine_qwen.json
"""
import argparse, gzip, json, math, random
from collections import defaultdict
from pathlib import Path

CONSERVATIVE, LENIENT = 0.5, 0.1     # SWORDS thresholds, as in eval_swords.py

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--swords", required=True)
p.add_argument("--out", required=True)
p.add_argument("--fake", action="store_true", help="random embeddings; exercises the code path")
a = p.parse_args()

# ---- SWORDS: per target, (candidate, human score) --------------------------------
with gzip.open(a.swords, "rt", encoding="utf-8") as h:
    data = json.load(h)
cands = defaultdict(list)
for sid, sub in data["substitutes"].items():
    labels = data["substitute_labels"][sid]
    if labels:
        cands[sub["target_id"]].append((sub["substitute"], sum(x == "TRUE" for x in labels) / len(labels)))
targets = {tid: t["target"] for tid, t in data["targets"].items()}

# ---- embeddings ----------------------------------------------------------------
if a.fake:
    import numpy as np
    rng = np.random.default_rng(0)
    table = {}
    def vec(word):
        return table.setdefault(word, rng.standard_normal(16))
    def single_token(word):
        return " " not in word.strip()
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    E = model.get_input_embeddings().weight.detach().float()
    del model
    def _ids(word):
        # Zhang's extractor scores whole-word tokens; the leading-space form is the one
        # that occurs mid-sentence, so prefer it and fall back to the bare form.
        for form in (" " + word, word):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                return ids[0]
        return None
    def single_token(word):
        return _ids(word) is not None
    def vec(word):
        return E[_ids(word)].numpy()

def cosine(u, v):
    import numpy as np
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

def auroc(pos, neg):
    # Mann-Whitney: P(score(pos) > score(neg)), ties count half.
    wins = sum((1.0 if x > y else 0.5 if x == y else 0.0) for x in pos for y in neg)
    return wins / (len(pos) * len(neg))

def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return None if r is None or math.isnan(r) else float(r)

# ---- within-target statistics, mirroring how SWORDS ranking metrics are computed --
per_target, n_cand, n_scored = [], 0, 0
for tid, word in targets.items():
    if not single_token(word):
        continue
    rows = [(c, h) for c, h in cands.get(tid, []) if single_token(c)]
    n_cand += len(cands.get(tid, [])); n_scored += len(rows)
    if len(rows) < 2:
        continue
    tv = vec(word)
    scores = [cosine(tv, vec(c)) for c, _ in rows]
    human = [h for _, h in rows]
    acc = [s for s, h in zip(scores, human) if h >= CONSERVATIVE]
    rej = [s for s, h in zip(scores, human) if h < LENIENT]
    rec = {"target_id": tid, "n": len(rows), "spearman": spearman(scores, human)}
    if acc and rej:
        rec["auroc"] = auroc(acc, rej)
    per_target.append(rec)

def mean(key):
    vals = [r[key] for r in per_target if r.get(key) is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

summary = {
    "model": a.model, "fake": a.fake,
    "targets_scored": len(per_target),
    "candidate_coverage": round(n_scored / max(n_cand, 1), 3),
    "within_target_auroc": mean("auroc"),
    "within_target_spearman": mean("spearman"),
    "reference_pretrained_model_auroc_left": 0.66,
}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"summary": summary, "per_target": per_target}, indent=1))
print(json.dumps(summary, indent=2))
verdict = summary["within_target_auroc"][0]
if verdict is not None:
    print("\nverdict:", "cosine carries acceptability signal -- the graded-target arm is worth training"
          if verdict >= 0.58 else
          "cosine is near chance -- no self-supervised within-set target exists; write, do not train")
