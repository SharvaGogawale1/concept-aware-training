#!/usr/bin/env python3
"""Iyer et al.'s downstream datasets, scored with Zhang et al.'s protocol.

Zhang et al. evaluate classification with MTEB on frozen, mean-pooled, L2-normalised
last-layer embeddings.  This script applies that protocol, reproduced line for line
from mteb 2.20.11 (AbsTaskClassification), to the datasets of Iyer et al. that load
from Hugging Face:

  GLUE (7 classification tasks; the GLUE column is their unweighted mean):
      CoLA, SST-2, MRPC, QQP, MNLI-matched, QNLI, RTE  -- scored on validation,
      because GLUE's test labels are hidden.
  SNLI, SPAM (SpamAssassin; its parquet conversion, pinned), HATE (Davidson et al.), FAKE (PolitiFact), LOG
  (logical fallacy).  EmpatheticDialogues is left out: it ships only as a loader
  script that current `datasets` cannot run.

MTEB's procedure, per task: shuffle the training indices with
np.random.RandomState(seed), keep the first `samples_per_label` rows of each label,
fit LogisticRegression(max_iter=100, random_state=seed) on their embeddings, predict
the evaluation split, and repeat for 10 experiments, each reshuffling the previous
order.  Scores are the mean over experiments.  MTEB's default budget is 8 examples
per label, which is what "Zhang's protocol" means here; 64 per label is also
reported as a lower-variance check.  Two-text tasks are embedded as one text,
"first\\nsecond", because MTEB's classification task embeds a single text.

Every split is pinned to a dataset revision.  Training-only datasets (SPAM, HATE)
get one fixed stratified 80/20 split; QQP's 40k validation rows are subsampled to
10k with a fixed seed.  A fingerprint of every evaluation set is stored, and a
resumed run refuses to mix evaluation sets.  Per-item correctness (how many of the
10 experiments got the item right) is kept for paired bootstrap tests.

    export HF_HUB_CACHE=$PWD/concept_aware/hf_cache/hub
    python scripts/eval_classification_probe.py --base-model meta-llama/Llama-3.2-1B \
        --manifests outputs/llama-3.2-1b/run_manifests/task15.json outputs/llama-3.2-1b/run_manifests/task15b.json \
        --arms pretrained zhang_seed42 zhang_plus_uniform_g0.0625_seed42 \
        --out outputs/llama-3.2-1b/classification_probe.json
"""
import argparse, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_encoder import FrozenEncoder, resolve_checkpoints

GLUE = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
TASKS = {
    # name: (repo, config, revision, train split, eval split, text columns, label column)
    "cola": ("nyu-mll/glue", "cola", GLUE, "train", "validation", ("sentence",), "label"),
    "sst2": ("nyu-mll/glue", "sst2", GLUE, "train", "validation", ("sentence",), "label"),
    "mrpc": ("nyu-mll/glue", "mrpc", GLUE, "train", "validation", ("sentence1", "sentence2"), "label"),
    "qqp": ("nyu-mll/glue", "qqp", GLUE, "train", "validation", ("question1", "question2"), "label"),
    "mnli": ("nyu-mll/glue", "mnli", GLUE, "train", "validation_matched", ("premise", "hypothesis"), "label"),
    "qnli": ("nyu-mll/glue", "qnli", GLUE, "train", "validation", ("question", "sentence"), "label"),
    "rte": ("nyu-mll/glue", "rte", GLUE, "train", "validation", ("sentence1", "sentence2"), "label"),
    "snli": ("stanfordnlp/snli", "plain_text", "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
             "train", "test", ("premise", "hypothesis"), "label"),
    # The repo itself is a loader script, which datasets>=4 refuses to run; Hugging
    # Face's parquet conversion of the same data is pinned by its commit instead.
    "spam": ("parquet", {"train": "hf://datasets/talby/spamassassin@1304e521a480006cf7e09cf54addb9206a763666"
                                  "/text/train/0000.parquet"}, None, "train", None, ("text",), "label"),
    "hate": ("tdavidson/hate_speech_offensive", None, "adc5fb774614827695774f2dbe0ea8122f6a92b4",
             "train", None, ("tweet",), "class"),
    "fake": ("Cartinoe5930/Politifact_fake_news", None, "d95d43ab9a96984e9aec4bbaf11d09f751cba013",
             "train", "test", ("news",), "label"),
    "logic": ("tasksource/logical-fallacy", None, "37e9b0537a86e72e9eaf6ee8c9a27d872a944103",
              "train", "test", ("source_article",), "logical_fallacies"),
}
GLUE_TASKS = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]
SEED, N_EXPERIMENTS, MAX_EVAL, MAX_TRAIN_POOL = 42, 10, 10_000, 50_000


def _texts(ds, cols):
    return ["\n".join(str(ds[c][i]) for c in cols) for i in range(len(ds))] if len(cols) > 1 \
        else [str(x) for x in ds[cols[0]]]


def load_task(name, limit=None):
    from datasets import load_dataset
    repo, config, rev, train_split, eval_split, cols, label_col = TASKS[name]
    ds = (load_dataset("parquet", data_files=config) if repo == "parquet"
          else load_dataset(repo, config, revision=rev))
    train = ds[train_split]
    train = train.filter(lambda ex: ex[label_col] is not None and ex[label_col] != -1)
    rng = np.random.RandomState(0)
    if eval_split is None:                              # fixed stratified 80/20
        labels = np.array(train[label_col])
        tr_idx, ev_idx = train_test_split(np.arange(len(train)), test_size=0.2, random_state=0,
                                          stratify=labels)
        train, evalset = train.select(sorted(tr_idx)), train.select(sorted(ev_idx))
    else:
        evalset = ds[eval_split].filter(lambda ex: ex[label_col] is not None and ex[label_col] != -1)
    if len(evalset) > MAX_EVAL:
        evalset = evalset.select(sorted(rng.choice(len(evalset), MAX_EVAL, replace=False)))
    if len(train) > MAX_TRAIN_POOL:                     # sampling pool only; MTEB draws <=64/label
        train = train.select(sorted(rng.choice(len(train), MAX_TRAIN_POOL, replace=False)))
    if limit:                                           # smoke test: random rows (files can be label-sorted)
        pick = lambda d: d.select(sorted(rng.choice(len(d), min(limit, len(d)), replace=False)))
        train, evalset = pick(train), pick(evalset)
    names = sorted({str(x) for x in train[label_col]} | {str(x) for x in evalset[label_col]})
    to_id = {n: i for i, n in enumerate(names)}
    task = {"train_text": _texts(train, cols), "train_y": [to_id[str(x)] for x in train[label_col]],
            "eval_text": _texts(evalset, cols), "eval_y": [to_id[str(x)] for x in evalset[label_col]],
            "labels": names}
    h = hashlib.sha256()
    for t, y in zip(task["eval_text"], task["eval_y"]):
        h.update(f"{y}\t{t}\n".encode())
    task["eval_fingerprint"] = h.hexdigest()[:16]
    return task


def undersample(labels, samples_per_label, idxs):
    """mteb AbsTaskClassification._undersample_data, verbatim in behaviour."""
    rng_state = np.random.RandomState(SEED)
    rng_state.shuffle(idxs)
    counter, picked = defaultdict(int), []
    for i in idxs:
        if counter[labels[i]] < samples_per_label:
            picked.append(i)
            counter[labels[i]] += 1
    return picked


def experiment_draws(labels, samples_per_label):
    idxs, draws = list(range(len(labels))), []
    for _ in range(N_EXPERIMENTS):
        draws.append(undersample(labels, samples_per_label, idxs))
    return draws


def probe(encoder, task, budgets):
    draws = {b: experiment_draws(task["train_y"], b) for b in budgets}
    union = sorted({i for per in draws.values() for d in per for i in d})
    train_emb = dict(zip(union, encoder.encode([task["train_text"][i] for i in union])))
    eval_emb = encoder.encode(task["eval_text"])
    y_eval = np.array(task["eval_y"])
    out = {}
    for b, per in draws.items():
        accs, f1s, correct = [], [], np.zeros(len(y_eval), dtype=int)
        for picked in per:
            clf = LogisticRegression(max_iter=100, random_state=SEED)
            clf.fit(np.stack([train_emb[i] for i in picked]), [task["train_y"][i] for i in picked])
            pred = clf.predict(eval_emb)
            accs.append(accuracy_score(y_eval, pred))
            f1s.append(f1_score(y_eval, pred, average="macro"))
            correct += (pred == y_eval)
        out[f"k{b}"] = {"accuracy": float(np.mean(accs)), "f1_macro": float(np.mean(f1s)),
                        "accuracy_sd_over_experiments": float(np.std(accs)),
                        "per_item_correct_of_10": correct.tolist()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--manifests", nargs="*", default=[])
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--samples-per-label", type=int, nargs="+", default=[8, 64],
                   help="8 = MTEB's default (Zhang's protocol); 64 = lower-variance check")
    p.add_argument("--limit", type=int, default=None, help="smoke test: rows per split")
    p.add_argument("--no-quantize", action="store_true", help="bf16/fp32 instead of 4-bit (CPU smoke tests)")
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    out_path = Path(a.out)
    if a.limit:
        out_path = out_path.with_name(f"{out_path.stem}_smoke{out_path.suffix}")
    results = json.loads(out_path.read_text()) if out_path.is_file() else {}
    tasks = {}
    for name in a.tasks:
        t0 = time.time()
        tasks[name] = load_task(name, a.limit)
        print(f"loaded {name}: {len(tasks[name]['train_y'])} train pool, {len(tasks[name]['eval_y'])} eval, "
              f"{len(tasks[name]['labels'])} labels ({time.time() - t0:.0f}s)")
    meta = results.setdefault("_meta", {"protocol": "mteb-2.20.11 logistic-regression probe",
                                        "seed": SEED, "experiments": N_EXPERIMENTS})
    for name, t in tasks.items():
        known = meta.setdefault("eval_fingerprints", {}).get(name)
        if known and known != t["eval_fingerprint"]:
            raise SystemExit(f"{name}: evaluation set differs from the one already in {out_path}")
        meta["eval_fingerprints"][name] = t["eval_fingerprint"]
        meta.setdefault("labels", {})[name] = t["labels"]

    for label, adapter in resolve_checkpoints(a.arms, a.manifests).items():
        done = results.get(label, {})
        todo = [n for n in tasks if n not in done]
        if not todo:
            print(f"resume: {label} already scored"); continue
        started = time.time()
        enc = FrozenEncoder(a.base_model, adapter, quantize=not a.no_quantize, batch_size=a.batch_size)
        for name in todo:
            done[name] = probe(enc, tasks[name], a.samples_per_label)
            results[label] = {**done, "_adapter": adapter}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results))
        enc.close()
        for b in a.samples_per_label:
            glue = [done[t][f"k{b}"]["accuracy"] for t in GLUE_TASKS if t in done]
            if glue:
                done[f"glue_mean_k{b}"] = float(np.mean(glue))
        results[label] = {**done, "_adapter": adapter, "_seconds": round(time.time() - started, 1)}
        out_path.write_text(json.dumps(results))
        k = f"k{a.samples_per_label[0]}"
        print(f"{label} ({results[label]['_seconds']}s): " + "  ".join(
            f"{n} {done[n][k]['accuracy']:.3f}" for n in tasks))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
