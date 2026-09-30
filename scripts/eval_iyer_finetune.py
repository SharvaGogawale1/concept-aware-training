#!/usr/bin/env python3
"""Iyer et al.'s downstream protocol: a short LoRA fine-tune per task, scored by generation.

The main tables score Iyer's datasets with Zhang's frozen probe
(eval_classification_probe.py).  This is the appendix comparability check: the same
pinned datasets and splits, fine-tuned as Iyer et al. (2026), Appendix B, describe:

  LoRA r=16, alpha=16 on q/k/v/o/gate/up/down_proj; 4-bit base; AdamW 8-bit;
  lr 2e-4, linear schedule; batch 2 x gradient accumulation 4; 100 steps;
  Alpaca instruction format; validation every 20 steps, best checkpoint kept;
  accuracy = lowercased, stripped generated answer == gold label.

Details the paper does not give, taken from the Unsloth Alpaca recipe that the listed
settings match: 5 warmup steps, weight decay 0.01, LoRA dropout 0, loss on the whole
formatted example (prompt + answer + EOS).  Instructions and label words are ours.
Their "GLUE" column appears to be MNLI (the appendix's GLUE examples and labels are MNLI's), so
MNLI is scored first; the other six GLUE tasks follow and give the 7-task mean used by
the probe table.

The post-trained checkpoint is the 4-bit base with its concept adapter kept frozen and
active; the task LoRA is a second adapter on top, so the concept weights are exactly
the ones every other evaluation scores (no merge and re-quantisation, which could wash
out a small adapter).  Compute is bf16 rather than the evaluations' fp16, for stable
training; this is the same for every arm.  "pretrained" gets the task LoRA alone.  The task LoRA's
initialisation, the 800 training and 200 validation examples, and the evaluation
subset are identical for every arm.

Besides Iyer's match accuracy, every item is also scored by ranking the label words
by log-likelihood (no unparseable answers), and per-item results are stored for the
paired bootstrap printed at the end.  Loops task-major, so an interrupted run leaves
complete arm comparisons for the tasks it finished; a rerun resumes.

    export HF_HOME=$PWD/concept_aware/hf_cache
    python scripts/eval_iyer_finetune.py --base-model meta-llama/Llama-3.2-1B \
        --manifests outputs/llama-3.2-1b/run_manifests/task15.json outputs/llama-3.2-1b/run_manifests/task15b.json \
        --arms zhang_seed42 zhang_plus_uniform_g0.0625_seed42 zhang_plus_uniform_g0.125_seed42 \
        --out outputs/llama-3.2-1b/iyer_finetune.json
"""
import argparse, gc, hashlib, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_classification_probe as probe
from frozen_encoder import resolve_checkpoints

SEED = 42
N_TRAIN, N_VAL, MAX_EVAL = 100 * 2 * 4, 200, 2000
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_INPUT_TOKENS = 384                                 # per example, split across its fields
ALPACA = ("Below is an instruction that describes a task, paired with an input that provides "
          "further context. Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")

NLI3 = {"0": "entailment", "1": "neutral", "2": "contradiction"}
NLI2 = {"0": "entailment", "1": "not entailment"}
# task: (instruction, field names shown in the input, raw label -> answer word)
SPECS = {
    "mnli": ("Does the premise entail the hypothesis? Answer entailment, neutral, or contradiction.",
             ("Premise", "Hypothesis"), NLI3),
    "snli": ("Does the premise entail the hypothesis? Answer entailment, neutral, or contradiction.",
             ("Premise", "Hypothesis"), NLI3),
    "hate": ("Classify the tweet. Answer hate speech, offensive language, or neither.",
             (None,), {"0": "hate speech", "1": "offensive language", "2": "neither"}),
    # ClassLabel(names=['spam', 'ham']) in the pinned parquet: 0 is spam.
    "spam": ("Is this email spam or ham? Answer spam or ham.", (None,), {"0": "spam", "1": "ham"}),
    # 0 = false, 1 = true (checked against the claims at the pinned revision).
    "fake": ("Is this political claim true or false? Answer true or false.", (None,),
             {"0": "false", "1": "true"}),
    "logic": (None, (None,), None),                   # instruction lists the 13 fallacies; labels are words
    "cola": ("Is this sentence grammatically acceptable? Answer acceptable or unacceptable.",
             (None,), {"0": "unacceptable", "1": "acceptable"}),
    "sst2": ("What is the sentiment of this sentence? Answer positive or negative.",
             (None,), {"0": "negative", "1": "positive"}),
    "mrpc": ("Do the two sentences mean the same thing? Answer equivalent or not equivalent.",
             ("Sentence 1", "Sentence 2"), {"0": "not equivalent", "1": "equivalent"}),
    "qqp": ("Do the two questions ask the same thing? Answer duplicate or not duplicate.",
            ("Question 1", "Question 2"), {"0": "not duplicate", "1": "duplicate"}),
    "qnli": ("Does the sentence answer the question? Answer entailment or not entailment.",
             ("Question", "Sentence"), NLI2),
    "rte": ("Does the first sentence entail the second? Answer entailment or not entailment.",
            ("Sentence 1", "Sentence 2"), NLI2),
}
TASK_ORDER = list(SPECS)                               # Iyer's six columns first
GLUE_TASKS = probe.GLUE_TASKS


def load(name):
    """The probe's pinned splits, with each example's text columns kept apart."""
    joined = probe._texts
    probe._texts = lambda ds, cols: [tuple(str(ds[c][i]) for c in cols) for i in range(len(ds))]
    try:
        t = probe.load_task(name)
    finally:
        probe._texts = joined
    instruction, fields, verbal = SPECS[name]
    words = [verbal[n] if verbal else n for n in t["labels"]]
    if instruction is None:
        instruction = "Which logical fallacy does the text contain? Answer one of: " + ", ".join(words) + "."
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(t["train_y"]))
    tr, va = order[:N_TRAIN], order[N_TRAIN:N_TRAIN + N_VAL]
    ev = np.arange(len(t["eval_y"]))
    if len(ev) > MAX_EVAL:
        ev = np.sort(np.random.RandomState(1).choice(len(ev), MAX_EVAL, replace=False))
    pick = lambda xs, ys, idx: ([xs[i] for i in idx], [ys[i] for i in idx])
    task = {"instruction": instruction, "fields": fields, "words": words,
            "train": pick(t["train_text"], t["train_y"], tr), "val": pick(t["train_text"], t["train_y"], va),
            "eval": pick(t["eval_text"], t["eval_y"], ev)}
    h = hashlib.sha256()
    for part in ("train", "val", "eval"):
        for x, y in zip(*task[part]):
            h.update(f"{part}\t{y}\t{x}\n".encode())
    task["fingerprint"] = h.hexdigest()[:16]
    return task


class Formatter:
    def __init__(self, tokenizer, task):
        self.tok, self.task = tokenizer, task

    def _clip(self, text):
        budget = MAX_INPUT_TOKENS // len(self.task["fields"])
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return text if len(ids) <= budget else self.tok.decode(ids[:budget])

    def prompt(self, x):
        parts = [self._clip(v) if f is None else f"{f}: {self._clip(v)}" for f, v in zip(self.task["fields"], x)]
        return ALPACA.format(instruction=self.task["instruction"], input="\n".join(parts))

    def encode(self, x, answer):
        """prompt + answer + EOS as ids, and where the answer starts."""
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None and \
            self.tok("a")["input_ids"][0] == self.tok.bos_token_id else []
        p = bos + self.tok(self.prompt(x), add_special_tokens=False)["input_ids"]
        a = self.tok(answer, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        return p + a, len(p)


def build_model(base_model, adapter, quantize, device):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    kwargs = {"device_map": {"": device}} if device.startswith("cuda") else {}   # on a Mac, device_map picks MPS
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if quantize:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                     target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    torch.manual_seed(SEED)                            # same task-LoRA init for every arm
    if adapter:
        peft = PeftModel.from_pretrained(model, adapter, adapter_name="concept", is_trainable=False)
        torch.manual_seed(SEED)
        peft.add_adapter("task", cfg)
        peft.base_model.set_adapter(["concept", "task"])   # both active; LoRA layers sum them
    else:
        peft = get_peft_model(model, cfg, adapter_name="task")
    for n, prm in peft.named_parameters():
        prm.requires_grad = ".task." in n and "lora_" in n
    # The injected LoRA layers carry the active adapters themselves, so the plain HF
    # model runs concept + task (PeftModel's own wrappers assume one active adapter).
    return peft, peft.get_base_model()


def batches(seqs, size, pad_id, left=False):
    for s in range(0, len(seqs), size):
        chunk = seqs[s:s + size]
        width = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, x in enumerate(chunk):
            sl = slice(width - len(x), width) if left else slice(0, len(x))
            ids[i, sl], mask[i, sl] = torch.tensor(x), 1
        yield ids, mask


def lm_loss(model, ids, mask, device):
    ids, mask = ids.to(device), mask.to(device)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    target = ids[:, 1:].masked_fill(mask[:, 1:] == 0, -100)
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                             ignore_index=-100)


def finetune(model, fmt, task, device, pad_id, log, steps=100):
    from transformers import get_linear_schedule_with_warmup
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=2e-4, weight_decay=0.01)
    except ImportError:                                 # CPU smoke tests only
        opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, min(5, steps - 1), steps)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    train = [fmt.encode(x, task["words"][y])[0] for x, y in zip(*task["train"])]
    val = [fmt.encode(x, task["words"][y])[0] for x, y in zip(*task["val"])]

    def val_loss():
        model.eval()
        with torch.no_grad(), amp:
            losses = [lm_loss(model, i, m, device).item() for i, m in batches(val, 4, pad_id)]
        model.train()
        return float(np.mean(losses))

    best, best_state, curve = math.inf, None, []
    model.train()
    micro = list(batches(train, 2, pad_id))
    every = max(1, steps // 5)
    for step in range(steps):
        for i, m in micro[step * 4:(step + 1) * 4]:
            with amp:
                (lm_loss(model, i, m, device) / 4).backward()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % every == 0:
            v = val_loss()
            curve.append(round(v, 4))
            if v < best:
                best = v
                best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n])
    model.eval()
    log(f"val loss every {every} steps {curve}, kept step {every * (curve.index(min(curve)) + 1)}")
    return curve


@torch.no_grad()
def evaluate(model, tok, fmt, task, device, pad_id, batch_size):
    words = task["words"]
    xs, ys = task["eval"]
    prompts = [fmt.encode(x, "")[0][:-1] for x in xs]    # prompt ids (drop the EOS of the empty answer)
    answer_len = max(len(tok(w, add_special_tokens=False)["input_ids"]) for w in words) + 2
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    generated = []
    for ids, mask in batches(prompts, batch_size, pad_id, left=True):
        with amp:
            out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                 max_new_tokens=answer_len, do_sample=False, pad_token_id=pad_id)
        generated += tok.batch_decode(out[:, ids.size(1):], skip_special_tokens=True)
    norm = lambda s: s.strip().split("\n")[0].strip().rstrip(".").strip().lower()
    match_pred = [norm(g) for g in generated]
    match_ok = [int(p == words[y].lower()) for p, y in zip(match_pred, ys)]
    to_id = {w.lower(): i for i, w in enumerate(words)}
    match_ids = [to_id.get(p, -1) for p in match_pred]

    # Ranked: sum log p(answer + EOS | prompt) for every label word.  The vocabulary
    # projection is applied only at answer positions (full logits would be ~9 GB a batch).
    decoder, head = model.get_decoder(), model.get_output_embeddings()
    pairs = [(i, k) for i in range(len(xs)) for k in range(len(words))]
    seqs = [fmt.encode(xs[i], words[k]) for i, k in pairs]
    scores = np.zeros((len(xs), len(words)))
    for s in range(0, len(seqs), batch_size):
        chunk = seqs[s:s + batch_size]
        ids, mask = next(batches([c[0] for c in chunk], len(chunk), pad_id))
        with amp:
            hidden = decoder(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
            for r, (seq, start) in enumerate(chunk):
                logp = head(hidden[r, start - 1:len(seq) - 1]).float().log_softmax(-1)
                tgt = torch.tensor(seq[start:], device=logp.device)
                scores[pairs[s + r]] = logp.gather(1, tgt[:, None]).sum().item()
    ranked = scores.argmax(1)
    y = np.array(ys)
    return {"match_accuracy": float(np.mean(match_ok)),
            "match_f1_macro": float(f1_score(y, match_ids, labels=list(range(len(words))), average="macro",
                                             zero_division=0)),
            "unparseable_rate": float(np.mean([m == -1 for m in match_ids])),
            "ranked_accuracy": float(np.mean(ranked == y)),
            "ranked_f1_macro": float(f1_score(y, ranked, average="macro")),
            "n_eval": len(ys), "per_item_match": match_ok, "per_item_ranked": (ranked == y).astype(int).tolist(),
            "sample_predictions": generated[:20]}


def paired(results, task, a, b, key, reps=10_000):
    x = np.array(results[task][b][key]) - np.array(results[task][a][key])
    rng = np.random.RandomState(SEED)
    boots = x[rng.randint(0, len(x), (reps, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--manifests", nargs="*", default=[])
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="*", default=TASK_ORDER)
    p.add_argument("--pair", action="append", default=[], help="A:B, reports B - A (default: first arm vs each other)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-quantize", action="store_true", help="fp32, no bitsandbytes (CPU smoke tests)")
    p.add_argument("--smoke", action="store_true", help="3 steps, 16 validation and 16 eval items")
    a = p.parse_args()
    global N_TRAIN, N_VAL, MAX_EVAL
    if a.smoke:
        N_TRAIN, N_VAL, MAX_EVAL = 3 * 2 * 4, 16, 16

    from transformers import AutoTokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    checkpoints = resolve_checkpoints(a.arms, a.manifests)
    out_path = Path(a.out)
    if a.smoke:
        out_path = out_path.with_name(f"{out_path.stem}_smoke{out_path.suffix}")
    results = json.loads(out_path.read_text()) if out_path.is_file() else {}
    meta = results.setdefault("_meta", {"protocol": "Iyer et al. Appendix B (LoRA fine-tune, generated answer)",
                                        "seed": SEED, "n_train": N_TRAIN, "n_val": N_VAL,
                                        "max_eval": MAX_EVAL, "base_model": a.base_model})
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    for name in a.tasks:
        todo = [arm for arm in checkpoints if arm not in results.get(name, {})]
        if not todo:
            log(f"resume: {name} done for every arm"); continue
        t0 = time.time()
        task = load(name)
        known = meta.setdefault("fingerprints", {}).get(name)
        if known and known != task["fingerprint"]:
            raise SystemExit(f"{name}: data differs from the run already in {out_path}")
        meta["fingerprints"][name] = task["fingerprint"]
        meta.setdefault("labels", {})[name] = task["words"]
        log(f"{name}: {len(task['train'][1])} train, {len(task['val'][1])} val, {len(task['eval'][1])} eval, "
            f"labels {task['words']} ({time.time() - t0:.0f}s)")
        fmt = Formatter(tok, task)
        for arm in todo:
            started = time.time()
            peft, model = build_model(a.base_model, checkpoints[arm], not a.no_quantize, device)
            curve = finetune(model, fmt, task, device, pad_id, log, steps=3 if a.smoke else 100)
            res = evaluate(model, tok, fmt, task, device, pad_id, a.batch_size)
            res.update({"val_curve": curve, "_adapter": checkpoints[arm], "_seconds": round(time.time() - started, 1)})
            results.setdefault(name, {})[arm] = res
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results))
            log(f"{name} {arm}: match {res['match_accuracy']:.3f} (unparseable {res['unparseable_rate']:.2f}), "
                f"ranked {res['ranked_accuracy']:.3f} ({res['_seconds']:.0f}s)")
            del peft, model
            gc.collect()                                # HF models hold reference cycles
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    arms = list(checkpoints)
    for arm in arms:
        glue = [results[t][arm]["match_accuracy"] for t in GLUE_TASKS if arm in results.get(t, {})]
        if len(glue) == len(GLUE_TASKS):
            meta.setdefault("glue_mean_match", {})[arm] = float(np.mean(glue))
    pairs = [x.split(":") for x in a.pair] or [(arms[0], b) for b in arms[1:]]
    for x, y in pairs:
        for name in a.tasks:
            if x in results.get(name, {}) and y in results.get(name, {}):
                for key in ("per_item_match", "per_item_ranked"):
                    d, lo, hi = paired(results, name, x, y, key)
                    meta.setdefault("paired", {}).setdefault(f"{y} - {x}", {}).setdefault(name, {})[key] = [d, lo, hi]
                d, lo, hi = meta["paired"][f"{y} - {x}"][name]["per_item_match"]
                log(f"{y} - {x}  {name}: match {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out_path.write_text(json.dumps(results))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
