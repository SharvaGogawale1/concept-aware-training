#!/usr/bin/env python3
"""Go/no-go for verified concept supervision: can an NLI verifier tell human-accepted
substitutes from rejected ones -- and can it do so BETTER than the model already does?

The proposal is to relabel the concept sets with an entailment verifier (original vs
substituted sentence, both directions; Omarov & Kondrak 2023) and to mine negatives that
are plausible in context but change meaning.  That only adds information the training
signal currently lacks if the verifier's judgement ranks SWORDS candidates within target
at least as well as the pretrained LM's own probability does -- .66 AUROC in left mode
for both Llama-3.2-1B and Qwen3-1.7B-Base -- and it is only usable at all if there exist
thresholds at which "verified positive" is mostly accepted and "verified negative" is
mostly rejected.  This measures both, on SWORDS dev, using the full sentence exactly as
the verifier would see it in the pipeline.

NLI models are known to over-predict entailment on near-identical sentences, which is
precisely the case here, so this cannot be assumed; it has to be read off the data.

    python scripts/diag_verifier_vs_human.py \
        --swords data/swords/swords-v1.1_dev.json.gz --out outputs/diag_verifier.json
"""
import argparse, gzip, json, math
from collections import defaultdict
from pathlib import Path

CONSERVATIVE, LENIENT = 0.5, 0.1     # SWORDS thresholds, as in eval_swords.py
LM_REFERENCE_AUROC = 0.66            # pretrained model, within-target, left mode

p = argparse.ArgumentParser()
p.add_argument("--swords", required=True)
p.add_argument("--out", required=True)
p.add_argument("--verifier", default="microsoft/deberta-large-mnli")
p.add_argument("--window", type=int, default=300, help="chars of context kept either side")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--fake", action="store_true", help="random scores; exercises the code path")
a = p.parse_args()

# ---- SWORDS: (original sentence, substituted sentence, human score) per candidate ----
with gzip.open(a.swords, "rt", encoding="utf-8") as h:
    data = json.load(h)
items = []   # (target_id, candidate, human, original, substituted)
for sid, sub in data["substitutes"].items():
    labels = data["substitute_labels"][sid]
    if not labels:
        continue
    t = data["targets"][sub["target_id"]]
    ctx, off, word = data["contexts"][t["context_id"]]["context"], int(t["offset"]), t["target"]
    if ctx[off:off + len(word)] != word:
        continue
    left = " ".join(ctx[max(0, off - a.window):off].split())
    right = " ".join(ctx[off + len(word):off + len(word) + a.window].split())
    human = sum(x == "TRUE" for x in labels) / len(labels)
    items.append((sub["target_id"], sub["substitute"], human,
                  f"{left} {word} {right}".strip(), f"{left} {sub['substitute']} {right}".strip()))
print(f"{len(items)} candidates over {len({i[0] for i in items})} targets")

# ---- verifier: P(entail) in both directions; meaning preserved = min of the two --------
if a.fake:
    import random
    random.seed(0)
    preserved = [random.random() for _ in items]
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.verifier)
    model = AutoModelForSequenceClassification.from_pretrained(a.verifier).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device == "cuda":
        model.half()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    ent = next(i for i, l in id2label.items() if "entail" in l)

    def p_entail(premises, hypotheses):
        out = []
        for i in range(0, len(premises), a.batch):
            enc = tok(premises[i:i + a.batch], hypotheses[i:i + a.batch], return_tensors="pt",
                      padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                probs = model(**enc).logits.float().softmax(-1)[:, ent]
            out.extend(probs.tolist())
            if (i // a.batch) % 50 == 0:
                print(f"  {i}/{len(premises)}", flush=True)
        return out

    orig = [it[3] for it in items]; subd = [it[4] for it in items]
    fwd = p_entail(orig, subd)          # original => substituted
    bwd = p_entail(subd, orig)          # substituted => original
    preserved = [min(f, b) for f, b in zip(fwd, bwd)]

# ---- within-target ranking quality, the way GAP is computed --------------------------
def auroc(pos, neg):
    wins = sum((1.0 if x > y else 0.5 if x == y else 0.0) for x in pos for y in neg)
    return wins / (len(pos) * len(neg))

by_target = defaultdict(list)
for it, s in zip(items, preserved):
    by_target[it[0]].append((s, it[2]))
aurocs, spearmans = [], []
from scipy.stats import spearmanr
for rows in by_target.values():
    if len(rows) < 2:
        continue
    s = [r[0] for r in rows]; h = [r[1] for r in rows]
    r = spearmanr(s, h).correlation
    if r is not None and not math.isnan(r):
        spearmans.append(float(r))
    acc = [x for x, y in rows if y >= CONSERVATIVE]; rej = [x for x, y in rows if y < LENIENT]
    if acc and rej:
        aurocs.append(auroc(acc, rej))

# ---- operating points: what the pipeline's thresholds would actually deliver --------
accepted = [s for it, s in zip(items, preserved) if it[2] >= CONSERVATIVE]
rejected = [s for it, s in zip(items, preserved) if it[2] < LENIENT]
middle = [s for it, s in zip(items, preserved) if LENIENT <= it[2] < CONSERVATIVE]
def operating(theta_pos, theta_neg):
    vp = [s >= theta_pos for s in accepted + rejected + middle]
    vn = [s <= theta_neg for s in accepted + rejected + middle]
    n_acc, n_rej = len(accepted), len(rejected)
    tp = sum(s >= theta_pos for s in accepted); fp = sum(s >= theta_pos for s in rejected)
    tn = sum(s <= theta_neg for s in rejected); fn = sum(s <= theta_neg for s in accepted)
    return {"theta_pos": theta_pos, "theta_neg": theta_neg,
            "verified_pos_precision_vs_rejected": tp / max(tp + fp, 1), "accepted_recall": tp / max(n_acc, 1),
            "verified_neg_precision_vs_accepted": tn / max(tn + fn, 1), "rejected_recall": tn / max(n_rej, 1),
            "abstain_share": 1 - (sum(vp) + sum(vn)) / max(len(vp), 1)}

points = [operating(tp_, tn_) for tp_, tn_ in ((0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4))]
summary = {
    "verifier": a.verifier, "fake": a.fake, "candidates": len(items),
    "targets_with_both_classes": len(aurocs),
    "within_target_auroc": sum(aurocs) / len(aurocs) if aurocs else None,
    "within_target_spearman": sum(spearmans) / len(spearmans) if spearmans else None,
    "lm_reference_auroc": LM_REFERENCE_AUROC,
    "mean_score": {"accepted": sum(accepted) / len(accepted), "rejected": sum(rejected) / len(rejected)},
    "operating_points": points,
}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"summary": summary,
                                   "per_candidate": [(it[0], it[1], it[2], round(s, 4)) for it, s in zip(items, preserved)]}, indent=1))
print(json.dumps(summary, indent=2))
au = summary["within_target_auroc"]
if au is not None:
    print("\nverdict:",
          "verifier ranks candidates BETTER than the LM does -- verified supervision can add information; build the pipeline"
          if au >= LM_REFERENCE_AUROC + 0.04 else
          "verifier is about as good as the LM already is -- it may still help as a filter, but do not expect a large gain"
          if au >= LM_REFERENCE_AUROC - 0.02 else
          "verifier ranks WORSE than the LM -- relabeling with it would remove information; stop here")
