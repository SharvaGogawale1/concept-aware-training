#!/usr/bin/env python3
"""Word similarity: MEN, WordSim-353 (all, similarity half, relatedness half), SimLex-999.

Each word is embedded on its own by the frozen model (scripts/frozen_encoder.py, the
STS representation), the cosine of the two word vectors is compared with the human
score, and the Spearman correlation is reported per dataset.  Two poolings are kept:
`words` averages the word's own tokens only (the reported number: a lone word is 1-3
tokens, so a BOS state would otherwise dominate) and `sts` averages every token
exactly as the STS evaluation does (a check that the choice does not drive results).

Files are fetched from one pinned commit of vecto-ai/word-benchmarks and checked by
SHA-256.  WordSim-353 is rebuilt as the union of Agirre et al.'s similarity and
relatedness halves, which is the original set with the original scores.  WS-353
lists money-cash twice, so the union is asserted to be its 352 distinct pairs.  Hypothesis stated before any run:
synonym-based concept training should move SimLex (similarity) and WS-353-sim more
than MEN and WS-353-rel (relatedness).

    python scripts/eval_word_similarity.py --base-model meta-llama/Llama-3.2-1B \
        --manifests outputs/llama-3.2-1b/run_manifests/task15.json outputs/llama-3.2-1b/run_manifests/task15b.json \
        --arms pretrained zhang_seed42 zhang_plus_uniform_g0.0625_seed42 \
        --out outputs/llama-3.2-1b/word_similarity.json
"""
import argparse, csv, hashlib, io, json, sys, time, urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_encoder import FrozenEncoder, resolve_checkpoints

COMMIT = "9b61d0e067e2e68af22160d09450f82d9544cd70"
BASE = f"https://raw.githubusercontent.com/vecto-ai/word-benchmarks/{COMMIT}/word-similarity/monolingual/en/"
FILES = {
    "men.csv": "ee7efb13c361afe12a6b1a58654833f97029e0b0540859f8db503c2f75782e44",
    "simlex999.csv": "44d0b1dd5d57706acdb6cd37bd5afa4e091d7552671b9d6d4a0f4bdc321c1792",
    "wordsim353-sim.csv": "97444f54a2c22693f7f458c15278976c125360f33f2f6f8e1c6245123c2c14c7",
    "wordsim353-rel.csv": "a20580f54dca51d02585568eaa6f4ba4d289498f6ecfa589eea46bf836885e76",
}


def fetch(name, cache):
    path = cache / name
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(BASE + name, timeout=60).read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FILES[name]:
        raise SystemExit(f"{path}: sha256 {digest} != pinned {FILES[name]}")
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    # wordsim353-sim.csv ends with an empty row ("203,,,"); skip rows without a pair.
    return [(r["word1"], r["word2"], float(r["similarity"])) for r in rows
            if r["word1"] and r["word2"] and r["similarity"]]


def load_benchmarks(cache):
    strip = lambda w: w.rsplit("-", 1)[0] if w[-2:] in ("-n", "-v", "-j") else w   # MEN's POS tags
    men = [(strip(a), strip(b), s) for a, b, s in fetch("men.csv", cache)]
    ws_sim, ws_rel = fetch("wordsim353-sim.csv", cache), fetch("wordsim353-rel.csv", cache)
    union = {}
    for a, b, s in ws_sim + ws_rel:
        key = (a.lower(), b.lower())
        if key in union and abs(union[key][2] - s) > 1e-6:
            raise SystemExit(f"WordSim halves disagree on {key}")
        union[key] = (a, b, s)
    ws_all = list(union.values())
    # The original WS-353 lists money-cash twice: 353 rows, 352 distinct pairs.
    assert len(ws_all) == 352, f"WordSim-353 union has {len(ws_all)} pairs, expected 352"
    return {"MEN": men, "WS353": ws_all, "WS353-sim": ws_sim, "WS353-rel": ws_rel,
            "SimLex999": fetch("simlex999.csv", cache)}


def score(encoder, benchmarks):
    words = sorted({w for pairs in benchmarks.values() for a, b, _ in pairs for w in (a, b)})
    out = {}
    for pooling, pool_special in (("words", False), ("sts", True)):
        vec = dict(zip(words, encoder.encode(words, pool_special=pool_special)))
        for name, pairs in benchmarks.items():
            cos = [float(vec[a] @ vec[b]) for a, b, _ in pairs]
            rho = spearmanr(cos, [s for *_, s in pairs]).correlation
            out.setdefault(name, {})[f"spearman_{pooling}"] = float(rho)
            out[name]["n"] = len(pairs)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--manifests", nargs="*", default=[])
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default="concept_aware/data/word_similarity")
    p.add_argument("--no-quantize", action="store_true", help="bf16/fp32 instead of 4-bit (CPU smoke tests)")
    p.add_argument("--batch-size", type=int, default=128)
    a = p.parse_args()

    benchmarks = load_benchmarks(Path(a.cache))
    checkpoints = resolve_checkpoints(a.arms, a.manifests)
    out_path = Path(a.out)
    results = json.loads(out_path.read_text()) if out_path.is_file() else {}
    for label, adapter in checkpoints.items():
        if label in results:
            print(f"resume: {label} already scored"); continue
        started = time.time()
        enc = FrozenEncoder(a.base_model, adapter, quantize=not a.no_quantize, batch_size=a.batch_size)
        results[label] = {**score(enc, benchmarks), "_adapter": adapter,
                          "_seconds": round(time.time() - started, 1)}
        enc.close()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{label}: " + "  ".join(f"{k} {v['spearman_words']:.3f}" for k, v in results[label].items()
                                         if isinstance(v, dict)))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
