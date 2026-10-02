#!/usr/bin/env python3
"""Did the concept term train?  Probability of the listed synonyms at the concept slot.

For every row of Iyer et al.'s loss files (text = the sentence up to the slot, plus its
synonym list), each checkpoint reads [BOS] + text and scores the next token.  Two numbers
per row, over the synonyms that are one token in context (" word"), which is what the
fixed objective supervises:
  set_logmass = log sum_c p(c | text)      (mass on the whole synonym set)
  mean_logp   = mean_c log p(c | text)     (Iyer's objective, the fixed arms' training target)
A loss arm whose concept term trained should beat its released (no_grad) twin on both by
a wide margin on the training file, and by some margin on the held-out validation file.
If the fixed arm only matches the released one, its concept term never reached the weights.

    python scripts/check_concept_slot.py --manifest outputs/llama-3.2-1b/run_manifests/iyer_replica.json \
        --data concept_aware/laya-concept-aware-training/data/syn/youtube --out outputs/llama-3.2-1b/concept_slot_check.json
"""
import argparse, ast, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PAIRS = [("iyer_nsp_loss_ctx", "iyer_nsp_loss_ctx_fixed"), ("iyer_nsp_loss_dict", "iyer_nsp_loss_dict_fixed")]


def rows(path, tok):
    df = pd.read_csv(path)
    col = "context_syn" if "context_syn" in df.columns else "dict_syn"
    out = []
    for text, syns in zip(df["text"], df[col]):
        try:
            words = ast.literal_eval(syns)
        except (ValueError, SyntaxError):
            continue
        ids = []
        for w in words:
            enc = tok(" " + str(w).strip(), add_special_tokens=False)["input_ids"]
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        if ids and isinstance(text, str) and "<mask>" not in text:
            out.append((tok(text)["input_ids"], ids))         # default special tokens: [BOS] + text
    return out


@torch.no_grad()
def score(model, data, device, batch=32):
    mass, mean = [], []
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    res = [None] * len(data)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        width = max(len(data[i][0]) for i in idx)
        ids = torch.zeros(len(idx), width, dtype=torch.long)
        att = torch.zeros(len(idx), width, dtype=torch.long)
        for r, i in enumerate(idx):
            x = data[i][0]
            ids[r, :len(x)], att[r, :len(x)] = torch.tensor(x), 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        last = att.sum(1) - 1
        lp = F.log_softmax(logits[torch.arange(len(idx)), last.to(device)].float(), -1)
        for r, i in enumerate(idx):
            c = lp[r, data[i][1]]
            res[i] = (torch.logsumexp(c, 0).item(), c.mean().item())
    return np.array(res)


def paired(a, b, reps=10_000):
    d = b - a
    boots = d[np.random.RandomState(0).randint(0, len(d), (reps, len(d)))].mean(1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data", required=True, help="Iyer's data/syn/youtube directory")
    p.add_argument("--base-model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--files", nargs="+", default=["context_loss_train.csv", "context_loss_val.csv",
                                                  "dict_loss_train.csv", "dict_loss_val.csv"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(a.base_model)   # every checkpoint shares these ids; [PAD] was appended
    arms = {"pretrained": a.base_model, **json.loads(Path(a.manifest).read_text())}
    data = {f: rows(Path(a.data) / f, tok) for f in a.files}
    for f, d in data.items():
        print(f"{f}: {len(d)} slots with at least one one-token synonym", flush=True)
    scores = {}
    for arm, path in arms.items():
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()
        scores[arm] = {f: score(model, d, device) for f, d in data.items()}
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"scored {arm}", flush=True)

    out = {"files": {f: len(d) for f, d in data.items()}, "mean": {}, "paired": {}}
    for f in a.files:
        print(f"\n{f}   (nats; higher = more probability on the synonyms)")
        print(f"  {'arm':28s} {'log P(set)':>11s} {'mean log p':>11s}")
        for arm in arms:
            m = scores[arm][f].mean(0)
            out["mean"].setdefault(f, {})[arm] = m.tolist()
            print(f"  {arm:28s} {m[0]:11.3f} {m[1]:11.3f}")
        for released, fixed in PAIRS:
            if released in scores and fixed in scores:
                for k, name in ((0, "log P(set)"), (1, "mean log p")):
                    d, lo, hi = paired(scores[released][f][:, k], scores[fixed][f][:, k])
                    out["paired"].setdefault(f, {}).setdefault(f"{fixed} - {released}", {})[name] = [d, lo, hi]
                    print(f"  {fixed} - {released}: {name} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
