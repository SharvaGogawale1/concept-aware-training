#!/usr/bin/env python3
"""Iyer et al.'s datasets with the language-model head replaced by a classification head.

eval_iyer_finetune.py fine-tunes each checkpoint to *generate* the label word.  This
script keeps everything about that run and changes only the output layer:
transformers' AutoModelForSequenceClassification drops the vocabulary projection
(lm_head, the next-token softmax) and puts a bias-free linear layer, hidden size ->
number of labels, on the final hidden state of the last real token.  Training is
cross-entropy over the labels, and the prediction is the argmax label, so nothing is
generated and no answer can be unparseable.

Shared with eval_iyer_finetune.py, so the two tables differ only in the head:
  same pinned datasets, same 800 training / 200 validation / <=2000 evaluation items,
  same input text (the Alpaca prompt up to "### Response:"), same budget:
  4-bit NF4 base with bf16 compute; the concept adapter frozen and active; task LoRA
  r=16, alpha=16 on q/k/v/o/gate/up/down_proj; AdamW 8-bit, lr 2e-4, linear schedule,
  5 warmup steps, weight decay 0.01; batch 2 x gradient accumulation 4; 100 steps;
  validation loss every 20 steps, best checkpoint kept.
The new head is trained in full (fp32) alongside the task LoRA, and is initialised
N(0, initializer_range) from a fixed seed, so every arm starts from the same head.

Reports accuracy and macro-F1 on the training items (after fine-tuning, with the kept
checkpoint), the validation items and the evaluation split, plus per-item test
correctness for the paired bootstrap printed at the end.  Task-major and resumable,
as the generative script is.

    export HF_HOME=$PWD/concept_aware/hf_cache
    python scripts/eval_seqcls_finetune.py --base-model meta-llama/Llama-3.2-1B \
        --manifests outputs/llama-3.2-1b/run_manifests/task15.json outputs/llama-3.2-1b/run_manifests/task15b.json \
        --arms pretrained zhang_seed42 zhang_plus_uniform_g0.0625_seed42 zhang_plus_uniform_g0.125_seed42 \
        --out outputs/llama-3.2-1b/seqcls_finetune.json
"""
import argparse, gc, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_iyer_finetune as iyer
from frozen_encoder import resolve_checkpoints

SEED = iyer.SEED


def build_model(base_model, adapter, n_labels, pad_id, quantize, device):
    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    if adapter and not (Path(adapter) / "adapter_config.json").is_file():
        base_model, adapter = adapter, None             # a fully fine-tuned model: it is the base
    kwargs = {"num_labels": n_labels, "pad_token_id": pad_id}
    if device.startswith("cuda"):
        kwargs["device_map"] = {"": device}
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["score"])          # the new head stays full precision
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(base_model, **kwargs)
    if quantize:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    head = model.score
    with torch.no_grad():                              # same head for every arm
        g = torch.Generator().manual_seed(SEED)
        w = torch.randn(head.weight.shape, generator=g) * model.config.initializer_range
        head.weight.copy_(w.to(head.weight.device, head.weight.dtype))
    cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none", target_modules=iyer.LORA_TARGETS)
    torch.manual_seed(SEED)                            # same task-LoRA init for every arm
    if adapter:
        # The concept adapters were saved as CAUSAL_LM, whose PEFT wrapper expects a
        # generative model.  Loading them with no task type uses the plain wrapper; the
        # module paths (model.layers.*) are the same in both heads, so every concept
        # weight lands where it did in training.
        concept_cfg = PeftConfig.from_pretrained(adapter)
        concept_cfg.task_type = None
        peft = PeftModel.from_pretrained(model, adapter, adapter_name="concept", is_trainable=False,
                                         config=concept_cfg)
        torch.manual_seed(SEED)
        peft.add_adapter("task", cfg)
        peft.base_model.set_adapter(["concept", "task"])   # both active; LoRA layers sum them
    else:
        peft = get_peft_model(model, cfg, adapter_name="task")
    hf = peft.get_base_model()
    for n, prm in hf.named_parameters():
        prm.requires_grad = (".task." in n and "lora_" in n) or n.startswith("score.")
    return peft, hf


def encode(fmt, x):
    """The generative run's prompt, without an answer or EOS."""
    return fmt.encode(x, "")[0][:-1]


def ce_loss(model, ids, mask, y, device):
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
    return torch.nn.functional.cross_entropy(logits, y.to(device)), logits


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
    xs, ys = task["train"]
    train = [encode(fmt, x) for x in xs]
    micro = [(ids, mask, torch.tensor(ys[s:s + 2])) for s, (ids, mask) in
             zip(range(0, len(train), 2), iyer.batches(train, 2, pad_id))]
    val = [encode(fmt, x) for x in task["val"][0]]
    val_y = task["val"][1]

    def val_loss():
        model.eval()
        losses = []
        with torch.no_grad(), amp:
            for s, (i, m) in zip(range(0, len(val), 8), iyer.batches(val, 8, pad_id)):
                y = torch.tensor(val_y[s:s + 8])
                losses.append(ce_loss(model, i, m, y, device)[0].item() * len(y))
        model.train()
        return float(np.sum(losses) / len(val_y))

    best, best_state, curve = math.inf, None, []
    model.train()
    every = max(1, steps // 5)
    for step in range(steps):
        for i, m, y in micro[step * 4:(step + 1) * 4]:
            with amp:
                (ce_loss(model, i, m, y, device)[0] / 4).backward()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % every == 0:
            v = val_loss()
            curve.append(round(v, 4))
            if v < best:
                best = v
                best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    if best_state is None:
        raise RuntimeError(f"validation loss was never finite: {curve}")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n])
    model.eval()
    log(f"val loss every {every} steps {curve}, kept step {every * (curve.index(min(curve)) + 1)}")
    return curve


@torch.no_grad()
def predict(model, fmt, xs, device, pad_id, batch_size):
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    seqs = [encode(fmt, x) for x in xs]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))   # length-sorted batches
    pred = np.zeros(len(seqs), dtype=int)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask = next(iyer.batches([seqs[i] for i in idx], len(idx), pad_id))
        with amp:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        pred[idx] = logits.float().argmax(-1).cpu().numpy()
    return pred


def score(pred, ys, n_labels):
    y = np.array(ys)
    return {"accuracy": float(np.mean(pred == y)),
            "f1_macro": float(f1_score(y, pred, labels=list(range(n_labels)), average="macro", zero_division=0)),
            "n": len(y)}


def main():
    global SEED
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--manifests", nargs="*", default=[])
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="*", default=iyer.TASK_ORDER)
    p.add_argument("--pair", action="append", default=[], help="A:B, reports B - A (default: first arm vs each other)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-quantize", action="store_true", help="fp32, no bitsandbytes (CPU smoke tests)")
    p.add_argument("--smoke", action="store_true", help="3 steps, 16 validation and 16 eval items")
    p.add_argument("--seed", type=int, default=SEED,
                   help="fine-tune seed: which training/validation examples, head and task-LoRA init, order.  "
                        "The evaluation items do not depend on it.  Use a separate --out per seed.")
    a = p.parse_args()
    SEED = iyer.SEED = a.seed
    if a.smoke:
        iyer.N_TRAIN, iyer.N_VAL, iyer.MAX_EVAL = 3 * 2 * 4, 16, 16

    from transformers import AutoTokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    checkpoints = resolve_checkpoints(a.arms, a.manifests, allow_full=True)
    out_path = Path(a.out)
    if a.smoke:
        out_path = out_path.with_name(f"{out_path.stem}_smoke{out_path.suffix}")
    results = json.loads(out_path.read_text()) if out_path.is_file() else {}
    meta = results.setdefault("_meta", {"protocol": "Iyer et al. Appendix B budget, LM head replaced by "
                                                    "AutoModelForSequenceClassification's linear head",
                                        "seed": SEED, "n_train": iyer.N_TRAIN, "n_val": iyer.N_VAL,
                                        "max_eval": iyer.MAX_EVAL, "base_model": a.base_model})
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    for name in a.tasks:
        todo = [arm for arm in checkpoints if arm not in results.get(name, {})]
        if not todo:
            log(f"resume: {name} done for every arm"); continue
        t0 = time.time()
        task = iyer.load(name)
        known = meta.setdefault("fingerprints", {}).get(name)
        if known and known != task["fingerprint"]:
            raise SystemExit(f"{name}: data differs from the run already in {out_path}")
        meta["fingerprints"][name] = task["fingerprint"]
        meta.setdefault("labels", {})[name] = task["words"]
        n_labels = len(task["words"])
        log(f"{name}: {len(task['train'][1])} train, {len(task['val'][1])} val, {len(task['eval'][1])} eval, "
            f"{n_labels} labels ({time.time() - t0:.0f}s)")
        fmt = iyer.Formatter(tok, task)
        for arm in todo:
            started = time.time()
            peft, model = build_model(a.base_model, checkpoints[arm], n_labels, pad_id, not a.no_quantize, device)
            curve = finetune(model, fmt, task, device, pad_id, log, steps=3 if a.smoke else 100)
            res = {}
            for part in ("train", "val", "eval"):
                xs, ys = task[part]
                pred = predict(model, fmt, xs, device, pad_id, a.batch_size)
                res[part] = score(pred, ys, n_labels)
                if part == "eval":
                    res["per_item_test"] = (pred == np.array(ys)).astype(int).tolist()
                    res["eval"]["pred_label_share"] = np.bincount(pred, minlength=n_labels).tolist()
            res.update({"val_curve": curve, "_adapter": checkpoints[arm], "_seconds": round(time.time() - started, 1)})
            results.setdefault(name, {})[arm] = res
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results))
            log(f"{name} {arm}: train {res['train']['accuracy']:.3f}, val {res['val']['accuracy']:.3f}, "
                f"test {res['eval']['accuracy']:.3f} (F1 {res['eval']['f1_macro']:.3f}) ({res['_seconds']:.0f}s)")
            del peft, model
            gc.collect()                                # HF models hold reference cycles
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    arms = list(checkpoints)
    for arm in arms:
        glue = [results[t][arm]["eval"]["accuracy"] for t in iyer.GLUE_TASKS if arm in results.get(t, {})]
        if len(glue) == len(iyer.GLUE_TASKS):
            meta.setdefault("glue_mean_test", {})[arm] = float(np.mean(glue))
    pairs = [x.split(":") for x in a.pair] or [(arms[0], b) for b in arms[1:]]
    for x, y in pairs:
        for name in a.tasks:
            if x in results.get(name, {}) and y in results.get(name, {}):
                d, lo, hi = iyer.paired(results, name, x, y, "per_item_test")
                meta.setdefault("paired", {}).setdefault(f"{y} - {x}", {})[name] = [d, lo, hi]
                log(f"{y} - {x}  {name}: test {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out_path.write_text(json.dumps(results))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
