#!/usr/bin/env python3
"""Evaluate whether accepted synonyms induce consistent hypernym predictions."""

import argparse
import gc
import json
import statistics
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from checkpoint_loading import load_causal_lm
from sequence_ncp_trainer import encode_candidate_continuation, sequence_log_probs_from_logits


TEMPLATES = (
    "In this context, {substitute} is a type of",
    "{substitute} and other",
    "A more general term for {substitute} is",
)


def js_divergence(log_p, log_q):
    p, q = log_p.exp(), log_q.exp()
    m = (p + q) / 2
    return float((0.5 * ((p * (log_p - m.log())).sum() + (q * (log_q - m.log())).sum())).cpu())


@torch.no_grad()
def score_candidates(model, tokenizer, context, candidates, block_size, device):
    encoded = [encode_candidate_continuation(tokenizer, context, c, block_size) for c in candidates]
    if any(item is None for item in encoded):
        return None
    width = max(len(item["input_ids"]) for item in encoded)
    ids = torch.tensor([
        item["input_ids"] + [tokenizer.pad_token_id] * (width - len(item["input_ids"]))
        for item in encoded
    ], device=device)
    mask = torch.tensor([
        item["attention_mask"] + [0] * (width - len(item["attention_mask"]))
        for item in encoded
    ], device=device)
    labels = torch.tensor([
        item["labels"] + [-100] * (width - len(item["labels"]))
        for item in encoded
    ], device=device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    return F.log_softmax(sequence_log_probs_from_logits(logits, labels).float(), dim=0)


def evaluate(model, tokenizer, rows, block_size, device):
    accepted_js, rejected_js = [], []
    per_template = defaultdict(lambda: {"accepted": [], "rejected": []})
    strata = defaultdict(list)
    evaluated_slots = 0
    for row in rows:
        base_context = row["input_sequence"]
        for slot in row.get("content_word_responses", row.get("content_words", [])):
            hypernyms = sorted({str(x).strip() for x in slot.get("hypernyms", []) if str(x).strip()})
            accepted = [slot.get("word", ""), *slot.get("synonyms", [])]
            accepted = [str(x).strip() for x in accepted if str(x).strip()]
            rejected = [str(x).strip() for x in slot.get("negatives", []) if str(x).strip()]
            if len(hypernyms) < 2 or len(accepted) < 2:
                continue
            evaluated_slots += 1
            template_accepted = []
            template_rejected = []
            for template_index, template in enumerate(TEMPLATES):
                distributions = []
                for substitute in accepted:
                    prompt = f"Context: {base_context}\n{template.format(substitute=substitute)}"
                    values = score_candidates(
                        model, tokenizer, prompt, [" " + h for h in hypernyms], block_size, device
                    )
                    if values is not None:
                        distributions.append(values)
                local = []
                for i in range(len(distributions)):
                    for j in range(i + 1, len(distributions)):
                        local.append(js_divergence(distributions[i], distributions[j]))
                if local:
                    template_accepted.extend(local)
                    per_template[template_index]["accepted"].extend(local)

                target_dist = distributions[0] if distributions else None
                if target_dist is not None:
                    for substitute in rejected:
                        prompt = f"Context: {base_context}\n{template.format(substitute=substitute)}"
                        values = score_candidates(
                            model, tokenizer, prompt, [" " + h for h in hypernyms], block_size, device
                        )
                        if values is not None:
                            value = js_divergence(target_dist, values)
                            template_rejected.append(value)
                            per_template[template_index]["rejected"].append(value)
            accepted_js.extend(template_accepted)
            rejected_js.extend(template_rejected)
            if template_accepted:
                key = (
                    str(slot.get("pos")),
                    "low" if (slot.get("target_frequency") or 0) <= 1 else "high",
                    "small" if (slot.get("concept_set_size") or len(accepted)) <= 2 else "large",
                    str(slot.get("wordnet_depth")),
                )
                strata[key].extend(template_accepted)

    def mean(values):
        return statistics.fmean(values) if values else None

    return {
        "evaluated_slots": evaluated_slots,
        "accepted_synonym_js": mean(accepted_js),
        "rejected_substitute_js": mean(rejected_js),
        "separation": (
            mean(rejected_js) - mean(accepted_js) if rejected_js and accepted_js else None
        ),
        "per_template": [
            {
                "template": TEMPLATES[i],
                "accepted_synonym_js": mean(per_template[i]["accepted"]),
                "rejected_substitute_js": mean(per_template[i]["rejected"]),
            }
            for i in range(len(TEMPLATES))
        ],
        "accepted_js_by_pos_frequency_setsize_depth": {
            "|".join(key): {"n": len(values), "mean": mean(values)}
            for key, values in sorted(strata.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--hierarchy_jsonl", required=True)
    parser.add_argument("--results_json", required=True)
    parser.add_argument("--block_size", type=int, default=256)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [json.loads(line) for line in open(args.hierarchy_jsonl) if line.strip()]
    results = []
    for checkpoint in args.checkpoints:
        model = load_causal_lm(checkpoint, dtype=dtype, device=device, base_model=args.base_model)
        results.append({"checkpoint": checkpoint, **evaluate(model, tokenizer, rows, args.block_size, device)})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with open(args.results_json, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
