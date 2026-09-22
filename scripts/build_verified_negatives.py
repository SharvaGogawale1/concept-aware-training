#!/usr/bin/env python3
"""Verified concept supervision: NLI-mined negatives, and optionally pruned positives.

Replaces the contextual-cosine ceiling in `build_contrastive_negatives.py`, which let
only tokenizer fragments through (in 23/31 trainable slots every negative was a letter
or word-prefix).  Here a candidate from the model's own top-k pool becomes a negative
when it is a real word of the right POS AND an entailment verifier says the
substituted sentence no longer means what the original meant.

The verifier's operating region was measured on SWORDS dev (`diag_verifier_vs_human`):
its LOW tail is clean -- min(P(entail) in both directions) <= .4 is 97% precise for
human-rejected substitutes -- while its high end is useless (80% of rejected candidates
also score >= .9, because NLI over-entails near-identical sentences).  So the verifier
is used only as a negative detector, and, with --prune-positives, to drop the few
listed alternatives that fall in the same low tail.  Nothing here grades positives.

Schema is preserved: every `content_word_responses` item gains `negatives` and
`negative_scores` (the verifier score, lower = more clearly meaning-changing); with
--prune-positives, `synonyms` is filtered and the dropped ones go to `synonyms_pruned`.

    python scripts/build_verified_negatives.py \
        --source .../embedding/synonyms_train.jsonl --topk '.../prompting/topk_*.jsonl' \
        --model Qwen/Qwen3-1.7B-Base --output .../synonyms_train_verified.jsonl \
        --report .../verified_negatives_report.json --sample .../verified_sample_50.csv \
        --prune-positives [--limit 50 for a pilot] [--start/--end for sharding]
"""
from __future__ import annotations
import argparse, csv, glob, json, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--topk", nargs="+", required=True, help="glob(s) of top-k shards")
p.add_argument("--model", required=True, help="tokenizer that produced `position`")
p.add_argument("--output", required=True)
p.add_argument("--report", required=True)
p.add_argument("--sample", default=None, help="stratified 50-row CSV to read by hand")
p.add_argument("--ext", default=None, help="path to learning-concepts/data (for the shared filters)")
p.add_argument("--verifier", default="microsoft/deberta-large-mnli")
p.add_argument("--theta", type=float, default=0.4, help="verified-negative ceiling on min P(entail)")
p.add_argument("--max-candidates", type=int, default=12, help="pool candidates verified per slot")
p.add_argument("--max-negatives", type=int, default=8)
p.add_argument("--prune-positives", action="store_true")
p.add_argument("--window", type=int, default=300)
p.add_argument("--max-length", type=int, default=256, help="must match extraction")
p.add_argument("--batch", type=int, default=128)
p.add_argument("--chunk-rows", type=int, default=100, help="rows scored per verifier call")
p.add_argument("--start", type=int, default=0); p.add_argument("--end", type=int, default=None)
p.add_argument("--limit", type=int, default=None, help="pilot: only this many rows")
p.add_argument("--fake-nli", action="store_true", help="random verifier scores (tests only)")
a = p.parse_args()

# ---- shared filters from the existing miner -----------------------------------------
ext = a.ext or next((str(c) for c in (Path("external/learning-concepts/data"),
                                       Path("concept_aware/learning-concepts/data")) if c.is_dir()), None)
assert ext, "pass --ext <learning-concepts/data>"
sys.path.insert(0, ext)
from build_contrastive_negatives import _normalise, _wordnet_form, _pos_counts, load_topk, POS_MAP  # noqa: E402
from nltk.corpus import wordnet as wn  # noqa: E402
from nltk.stem import PorterStemmer  # noqa: E402
stem = PorterStemmer().stem

from functools import lru_cache as _lru  # noqa: E402

@_lru(maxsize=200_000)
def _counts_cached(word: str):
    return _pos_counts(word)

@_lru(maxsize=200_000)
def _synsets_cached(form: str, pos: str):
    return frozenset(wn.synsets(form, pos=pos))

# Quantifiers and determiners that WordNet lists as adjectives or nouns and the LM pool
# offers freely ("in both ways", "a group of three").  Seen in the pilot sample; a slot
# never needs them as negatives, and they are not what a substitution benchmark rejects.
STOPLIST = {"both", "other", "such", "same", "several", "various", "many", "much", "few",
            "less", "more", "most", "each", "every", "another", "either", "neither",
            "some", "none", "half", "whole", "total", "entire", "single", "double",
            "first", "second", "third", "last", "next", "previous", "former", "latter",
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "hundred", "thousand", "million", "billion", "dozen", "hundreds",
            "thousands", "millions", "billions", "dozens", "lots", "plenty"}

def is_word_like(candidate: str, wn_pos: str, target: str) -> bool:
    """Alphabetic, >= 4 letters, target's capitalisation class, and the requested POS is
    the word's dominant WordNet POS -- zero SemCor counts allowed.  Looser than the
    strict filter (which required a count >= 2 and left 3/31 slots), because the
    verifier now carries the meaning test; this only has to keep fragments out."""
    s = str(candidate).strip()
    if not s.isalpha() or len(s) < 4 or s.lower() in STOPLIST:
        return False
    if s[0].isupper() and not str(target).strip()[:1].isupper():
        return False
    counts = _counts_cached(s)
    return counts[wn_pos] >= max(counts.values(), default=0)

# ---- tokenizer offsets: `position` -> character span ---------------------------------
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.model)

from functools import lru_cache  # noqa: E402

@lru_cache(maxsize=4096)
def _offsets(text: str):
    # One tokenization per document, not one per slot (~41 slots share a text).
    return tok(text, return_offsets_mapping=True, add_special_tokens=True,
               truncation=True, max_length=a.max_length)["offset_mapping"]

def span_of(text: str, position: int, word: str):
    off = _offsets(text)
    if 0 <= position < len(off):
        s, e = off[position]
        if text[s:e].strip().lower() == word.strip().lower():
            s += len(text[s:e]) - len(text[s:e].lstrip())      # drop a leading-space token prefix
            return s, e
    hits = [m.start() for m in re.finditer(rf"(?<!\w){re.escape(word)}(?!\w)", text)]
    return (hits[0], hits[0] + len(word)) if len(hits) == 1 else None

def single_token(word: str) -> bool:
    try:
        return len(tok.encode(" " + str(word).strip(), add_special_tokens=False)) == 1
    except Exception:
        return False

# ---- the verifier ---------------------------------------------------------------------
class Verifier:
    def __init__(self):
        self.cache = {}
        if a.fake_nli:
            self.rng = random.Random(0); return
        import torch
        from transformers import AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(a.verifier)
        self.model = AutoModelForSequenceClassification.from_pretrained(a.verifier).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if self.device == "cuda":
            self.model.half()
        labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ent = next(i for i, l in labels.items() if "entail" in l)

    def _entail(self, prem, hyp):
        out = []
        for i in range(0, len(prem), a.batch):
            enc = self.tok(prem[i:i + a.batch], hyp[i:i + a.batch], return_tensors="pt",
                           padding=True, truncation=True, max_length=512).to(self.device)
            with self.torch.no_grad():
                out.extend(self.model(**enc).logits.float().softmax(-1)[:, self.ent].tolist())
        return out

    def score(self, pairs):
        """min P(entail) over both directions for each (original, substituted); cached."""
        todo = [pr for pr in dict.fromkeys(pairs) if pr not in self.cache]
        if todo:
            if a.fake_nli:
                for pr in todo: self.cache[pr] = self.rng.random()
            else:
                o, s = [pr[0] for pr in todo], [pr[1] for pr in todo]
                fwd, bwd = self._entail(o, s), self._entail(s, o)
                for pr, f, b in zip(todo, fwd, bwd): self.cache[pr] = min(f, b)
        return [self.cache[pr] for pr in pairs]

verifier = Verifier()
topk_index = load_topk(a.topk)

# ---- per-slot candidate selection --------------------------------------------------------
def candidates_for(target, pool, reasons):
    wn_pos = POS_MAP.get(target.get("pos") or pool.get("pos"))
    if wn_pos is None:
        reasons["unsupported_pos"] += 1; return None, []
    positives = [target["word"], *target.get("synonyms", [])]
    pos_norm = {_normalise(w) for w in positives}
    pos_stems = {stem(w) for w in pos_norm}
    pos_synsets = {s for w in positives for s in _synsets_cached(_wordnet_form(w), wn_pos)}
    kept = []
    for cand in pool.get("topk_tokens", []):
        if len(kept) >= a.max_candidates:
            break
        clean = _normalise(cand)
        if not clean or clean in pos_norm:
            reasons["positive_or_empty"] += 1; continue
        if stem(clean) in pos_stems:
            reasons["morphological_variant"] += 1; continue
        if not is_word_like(clean, wn_pos, target["word"]):
            reasons["not_word_like"] += 1; continue
        cs = _synsets_cached(_wordnet_form(clean), wn_pos)
        if not cs:
            reasons["no_matching_wordnet_pos"] += 1; continue
        if cs & pos_synsets:
            reasons["shared_synset"] += 1; continue
        if clean in kept:
            continue
        kept.append(clean)
    return wn_pos, kept

# ---- main loop, chunked so the verifier sees big batches -----------------------------------
reasons = Counter()
stats = Counter()
sample_pool = defaultdict(list)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()

def flush(rows, dst):
    """Score every pending pair in `rows` in one verifier pass, then decide and write."""
    pairs = []
    for row in rows:
        for job in row["_jobs"]:
            pairs.append(job["pair"])
    scores = verifier.score(pairs) if pairs else []
    it = iter(scores)
    for row in rows:
        for job in row["_jobs"]:
            job["score"] = next(it)
        for target in row.get("content_word_responses", []):
            jobs = [j for j in row["_jobs"] if j["target"] is target]
            negs = [(j["cand"], j["score"]) for j in jobs if j["kind"] == "neg" and j["score"] <= a.theta]
            negs = negs[:a.max_negatives]
            target["negatives"] = [c for c, _ in negs]
            target["negative_scores"] = [round(s, 4) for _, s in negs]
            stats["candidates_verified"] += sum(j["kind"] == "neg" for j in jobs)
            stats["negatives_kept"] += len(negs)
            stats["slots_with_negatives"] += bool(negs)
            if a.prune_positives:
                pj = {j["cand"]: j["score"] for j in jobs if j["kind"] == "pos"}
                keep, drop = [], []
                for syn in target.get("synonyms", []):
                    (drop if pj.get(_normalise(syn), 1.0) <= a.theta else keep).append(syn)
                target["synonyms_pruned"] = [[s, round(pj[_normalise(s)], 4)] for s in drop]
                target["synonyms"] = keep
                stats["positives_seen"] += len(keep) + len(drop)
                stats["positives_pruned"] += len(drop)
            # What the TRAINER would keep: single-token alternative AND single-token negative.
            alt_ok = any(single_token(s) for s in target.get("synonyms", []))
            neg_ok = any(single_token(n) for n in target["negatives"])
            stats["slots_with_single_token_alternative"] += alt_ok
            stats["effective_slots"] += alt_ok and neg_ok
            if negs and a.sample:
                key = (target.get("pos", "?"), "small" if len(target.get("synonyms", [])) <= 2 else "large")
                sample_pool[key].append({
                    "pos": key[0], "set_size": key[1], "context": row["_ctx"].get(id(target), ""),
                    "observed_target": target["word"],
                    "alternatives": " | ".join(target.get("synonyms", [])),
                    "pruned": " | ".join(f"{s} ({v})" for s, v in target.get("synonyms_pruned", [])),
                    "negatives": " | ".join(f"{c} ({s:.3f})" for c, s in negs)})
        row.pop("_jobs"); row.pop("_ctx")
        dst.write(json.dumps(row) + "\n")

with open(a.source, encoding="utf-8") as src, open(a.output, "w", encoding="utf-8") as dst:
    pending = []
    for idx, line in enumerate(src):
        if idx < a.start: continue
        if a.end is not None and idx >= a.end: break
        if a.limit is not None and stats["rows"] >= a.limit: break
        row = json.loads(line); stats["rows"] += 1
        text = row["input_sequence"]
        row["_jobs"], row["_ctx"] = [], {}
        for target in row.get("content_word_responses", []):
            stats["slots"] += 1
            pool = topk_index.get((text, int(target["position"])))
            if pool is None:
                reasons["missing_topk_slot"] += 1; target["negatives"] = []; target["negative_scores"] = []; continue
            target.setdefault("pos", pool.get("pos"))
            span = span_of(text, int(target["position"]), target["word"])
            if span is None:
                reasons["offset_mismatch"] += 1; target["negatives"] = []; target["negative_scores"] = []; continue
            s, e = span
            left = " ".join(text[max(0, s - a.window):s].split())
            right = " ".join(text[e:e + a.window].split())
            orig = f"{left} {target['word']} {right}".strip()
            row["_ctx"][id(target)] = f"{left[-120:]} [{target['word']}] {right[:80]}"
            wn_pos, cands = candidates_for(target, pool, reasons)
            for c in cands:
                row["_jobs"].append({"target": target, "kind": "neg", "cand": c,
                                     "pair": (orig, f"{left} {c} {right}".strip())})
            if a.prune_positives:
                for syn in target.get("synonyms", []):
                    row["_jobs"].append({"target": target, "kind": "pos", "cand": _normalise(syn),
                                         "pair": (orig, f"{left} {syn} {right}".strip())})
        pending.append(row)
        if len(pending) >= a.chunk_rows:
            flush(pending, dst); pending = []
            done = stats["rows"]
            print(f"  rows {done}  pairs scored {len(verifier.cache)}  "
                  f"slots w/ negatives {stats['slots_with_negatives']}/{stats['slots']}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    if pending:
        flush(pending, dst)

# ---- report and hand-reading sample -----------------------------------------------------------------
report = {
    "source": a.source, "output": a.output, "model": a.model, "verifier": a.verifier,
    "theta": a.theta, "max_candidates": a.max_candidates, "max_negatives": a.max_negatives,
    "prune_positives": a.prune_positives, "fake_nli": a.fake_nli,
    "rows": stats["rows"], "slots": stats["slots"],
    "slots_with_negatives": stats["slots_with_negatives"],
    "negative_coverage": stats["slots_with_negatives"] / max(stats["slots"], 1),
    "candidates_verified": stats["candidates_verified"], "negatives_kept": stats["negatives_kept"],
    "negatives_per_covered_slot": stats["negatives_kept"] / max(stats["slots_with_negatives"], 1),
    "positives_seen": stats["positives_seen"], "positives_pruned": stats["positives_pruned"],
    "slots_with_single_token_alternative": stats["slots_with_single_token_alternative"],
    "effective_slots": stats["effective_slots"],
    "effective_contrastive_coverage": stats["effective_slots"] / max(stats["slots_with_single_token_alternative"], 1),
    "unique_pairs_scored": len(verifier.cache), "seconds": round(time.time() - t0, 1),
    "rejection_reasons": dict(reasons),
}
Path(a.report).parent.mkdir(parents=True, exist_ok=True)
Path(a.report).write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "rejection_reasons"}, indent=2))
print("rejections:", dict(reasons))

if a.sample:
    rng = random.Random(0); picked = []
    keys = sorted(sample_pool)
    while len(picked) < 50 and any(sample_pool[k] for k in keys):
        for k in keys:
            if sample_pool[k] and len(picked) < 50:
                picked.append(sample_pool[k].pop(rng.randrange(len(sample_pool[k]))))
    with open(a.sample, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["pos", "set_size", "context", "observed_target",
                                          "alternatives", "pruned", "negatives"])
        w.writeheader(); w.writerows(picked)
    print(f"wrote {len(picked)} rows across {len(keys)} strata to {a.sample}")
