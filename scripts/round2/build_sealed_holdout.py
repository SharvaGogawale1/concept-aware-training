"""Task 2: sealed CoInCo holdout IDs.  Builds IDs only; never loads a model or scores.

Pool = every CoInCo target (dev + test, as parsed by the SWORDS repo) that
  - is not in coinco_dev_sub800 / coinco_test_sub800 (the subsets used so far), and
  - does not sit in a context that appears in SWORDS dev or test.  "Appears" is two
    checks, both reported: the CoInCo ids SWORDS records for its targets
    (extra.coinco_ids, same split, same word -> that CoInCo context), and the CoInCo
    sentence occurring inside any SWORDS window (SWORDS windows span several sentences),
    compared after whitespace normalization and again with letters and digits only,
    because SWORDS re-detokenized the text (421 id-matched sentences are not verbatim).
Eligibility = the existing rules, unchanged:
  build_lexsub_benchmarks.build_coinco: >= 1 accepted and >= 1 rejected substitute
  eval_swords.load_swords:              offset aligned, >= 2 substitutes
  eval_swords.score_record + filter:    >= 2 non-target substitutes encodable, in both
                                        scorers, for EVERY tokenizer (intersection)
Sample = 2,000 targets, stratified by POS x lemma-frequency bucket, proportional
allocation (largest remainder), seeded, without replacement; all of them if fewer.

    python scripts/round2/build_sealed_holdout.py            # server: all three tokenizers
Writes sealed/sealed_holdout_ids.json and sealed/holdout_creation_report.md, and the
derived evaluation file results/round2/sealed_data/coinco_sealed.json.gz (not committed;
its SHA-256 is pinned in the IDs file).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import r2common as C

RAW = C.MAIN / "data/lexsub/raw"
USED = [C.MAIN / "data/lexsub/coinco_dev_sub800.json.gz", C.MAIN / "data/lexsub/coinco_test_sub800.json.gz"]
SWORDS = [C.MAIN / "data/swords/swords-v1.1_dev.json.gz", C.MAIN / "data/swords/swords-v1.1_test.json.gz"]
TARGET_N = 2000
SEED = 20261016
MIN_POOL = 500            # below this, stop and report rather than build a weak holdout
# Quartiles of the per-target (lemma, POS) count over all 15,415 CoInCo targets,
# computed before any sampling: 25% = 3, 50% = 8, 75% = 21.
FREQ_EDGES = (3, 8, 21)
FREQ_LABELS = ("1-3", "4-8", "9-21", "22+")
SPLIT_BUILD = C.MAIN / "builddataset"
EVAL_DIR = C.MAIN / "transformers/examples/pytorch/language-modeling"


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def norm(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def squash(text):
    """Letters and digits only: SWORDS re-detokenized CoInCo, so spacing and quotes differ."""
    return re.sub(r"[^0-9a-z]+", "", text.casefold())


def freq_bucket(count):
    return FREQ_LABELS[sum(count > e for e in FREQ_EDGES)]


def coinco_view(source, keep):
    """build_lexsub_benchmarks.build_coinco's conversion, for an explicit id list."""
    sys.path.insert(0, str(SPLIT_BUILD))
    by_target = defaultdict(list)
    for substitute_id, substitute in source["substitutes"].items():
        labels = source["substitute_labels"].get(substitute_id) or []
        if not labels:
            continue
        accepted = all(label.startswith("TRUE") for label in labels)
        rejected = all(label.startswith("FALSE") for label in labels)
        if not (accepted or rejected):
            raise ValueError(f"mixed labels on one substitute: {labels}")
        freq = 1
        for extra in substitute.get("extra", []):
            attrs = ((extra or {}).get("coinco") or {}).get("xml_attrs") or {} if isinstance(extra, dict) else {}
            if attrs.get("freq"):
                freq = int(attrs["freq"])
        by_target[substitute["target_id"]].append((substitute_id, substitute["substitute"],
                                                  freq if accepted else 0))
    data = {"contexts": {}, "targets": {}, "substitutes": {}, "substitute_labels": {}}
    for target_id in keep:
        target = source["targets"][target_id]
        cid = target["context_id"]
        data["contexts"][cid] = {"context": source["contexts"][cid]["context"]}
        data["targets"][target_id] = {"context_id": cid, "target": target["target"],
                                      "offset": int(target["offset"]), "pos": target.get("pos", "?")}
        for sid, sub, freq in by_target[target_id]:
            data["substitutes"][sid] = {"target_id": target_id, "substitute": sub}
            data["substitute_labels"][sid] = ["TRUE"] * freq if freq else ["FALSE"]
    return data, by_target


def write_json_gz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    path.write_bytes(buffer.getvalue())


def tokenizer_eligible(view_path, tokenizer_id):
    """Target ids eval_swords.py would score in BOTH modes with this tokenizer."""
    sys.path.insert(0, str(EVAL_DIR))
    from transformers import AutoTokenizer
    from eval_swords import load_swords
    from sequence_ncp_trainer import encode_candidate_continuation
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ok = set()
    for record in load_swords(str(view_path), None, 1200, None):
        fine = True
        for mode in ("left", "full"):
            usable = 0
            for sub in record["substitutes"]:
                if sub.casefold() == record["target"].casefold():
                    continue
                cand = sub if mode == "left" else f'{sub}{record["right"]}'
                if encode_candidate_continuation(tok, record["left"], cand, 256) is not None:
                    usable += 1
            fine &= usable >= 2
        if fine:
            ok.add(record["target_id"])
    return ok


def allocate(strata, n, rng):
    total = sum(len(v) for v in strata.values())
    if total <= n:
        return {k: sorted(v) for k, v in strata.items()}
    quota = {k: n * len(v) / total for k, v in strata.items()}
    base = {k: int(q) for k, q in quota.items()}
    rest = n - sum(base.values())
    for k in sorted(quota, key=lambda k: (-(quota[k] - base[k]), k))[:rest]:
        base[k] += 1
    return {k: sorted(rng.sample(sorted(v), base[k])) for k, v in sorted(strata.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizers", default=",".join(C.FAMILIES.values()))
    ap.add_argument("--n", type=int, default=TARGET_N)
    ap.add_argument("--force", action="store_true", help="rewrite an existing IDs file (only before FREEZE)")
    ap.add_argument("--out-dir", default=str(C.SEALED), help="dry runs only; the real IDs live in sealed/")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    data_dir = C.R2 / "sealed_data" if out_dir == C.SEALED else out_dir
    ids_path = out_dir / "sealed_holdout_ids.json"
    if ids_path.exists() and not args.force:
        sys.exit(f"{ids_path} exists; the holdout is drawn once.  --force only before FREEZE.md.")
    if (C.SEALED / "FREEZE.md").exists():
        sys.exit("FREEZE.md exists: the holdout IDs can no longer change.")

    print(f"[{C.now_utc()}] loading CoInCo, SWORDS and the used 800-target subsets", flush=True)
    sources = {split: load(RAW / f"coinco_{split}.json.gz") for split in ("dev", "test")}
    used = set()
    for path in USED:
        used |= set(load(path)["targets"])
    swords_contexts, swords_coinco = [], {"dev": set(), "test": set()}
    for path in SWORDS:
        sw = load(path)
        split = "dev" if "_dev" in path.name else "test"
        swords_contexts += [c["context"] for c in sw["contexts"].values()]
        for t in sw["targets"].values():
            # SWORDS dev/test were built from CoInCo dev/test; ids repeat across splits.
            swords_coinco[split] |= {(str(x), t["target"].casefold())
                                     for x in (t.get("extra") or {}).get("coinco_ids", [])}
    swords_blob = "\n".join(norm(c) for c in swords_contexts)
    swords_squashed = "\n".join(squash(c) for c in swords_contexts)

    lemma_count = Counter()
    for src in sources.values():
        for t in src["targets"].values():
            a = t["extra"][0]["coinco"]["xml_attrs"]
            lemma_count[(a["lemma"].casefold(), t["pos"])] += 1

    counts = Counter()
    pool = {}                                  # target_id -> (split, pos, bucket, lemma)
    for split, src in sources.items():
        ctx_in_swords = set()
        for cid, c in src["contexts"].items():
            text, flat = norm(c["context"]), squash(c["context"])
            if (text and text in swords_blob) or (len(flat) >= 20 and flat in swords_squashed):
                ctx_in_swords.add(cid)
        # Contexts of the CoInCo targets SWORDS names by id (same split, same word).
        # 421 of them are NOT verbatim inside their SWORDS window, so this check stands alone.
        named = {t["context_id"] for t in src["targets"].values()
                 if (str(t["extra"][0].get("legacy_id")), t["target"].casefold()) in swords_coinco[split]}
        _, by_target = coinco_view(src, [])
        for tid, t in src["targets"].items():
            counts[f"{split}: all"] += 1
            if tid in used:
                counts[f"{split}: excluded, in a used 800-subset"] += 1
                continue
            if t["context_id"] in named:
                counts[f"{split}: excluded, context of a SWORDS target (by CoInCo id)"] += 1
                continue
            if t["context_id"] in ctx_in_swords:
                counts[f"{split}: excluded, sentence inside a SWORDS context (text)"] += 1
                continue
            subs = by_target.get(tid, [])
            if not (any(f > 0 for *_, f in subs) and any(f == 0 for *_, f in subs)):
                counts[f"{split}: ineligible, needs >=1 accepted and >=1 rejected"] += 1
                continue
            ctx = src["contexts"][t["context_id"]]["context"]
            off = int(t["offset"])
            if ctx[off:off + len(t["target"])] != t["target"] or len(subs) < 2:
                counts[f"{split}: ineligible, offset/<2 substitutes"] += 1
                continue
            lemma = t["extra"][0]["coinco"]["xml_attrs"]["lemma"].casefold()
            pool[tid] = (split, t["pos"], freq_bucket(lemma_count[(lemma, t["pos"])]), lemma)
            counts[f"{split}: pool before tokenizer check"] += 1

    # Tokenizer eligibility, through the evaluator's own loader and encoder.
    tmp = data_dir / "_pool_view.json.gz"
    views = {}
    for split, src in sources.items():
        keep = sorted(t for t, v in pool.items() if v[0] == split)
        views[split], _ = coinco_view(src, keep)
    merged = {k: {**views["dev"][k], **views["test"][k]} for k in views["dev"]}
    write_json_gz(tmp, merged)
    per_model = {}
    eligible = set(pool)
    print(f"[{C.now_utc()}] pool before the tokenizer check: {len(pool)} targets", flush=True)
    for tok in args.tokenizers.split(","):
        print(f"[{C.now_utc()}] tokenizer check: {tok} (a few minutes each)", flush=True)
        ok = tokenizer_eligible(tmp, tok)
        per_model[tok] = {"eligible": len(ok & set(pool)), "excluded": sorted(set(pool) - ok)}
        eligible &= ok
    tmp.unlink()
    if len(eligible) < MIN_POOL:
        sys.exit(f"only {len(eligible)} eligible targets (< {MIN_POOL}); stop and report -- "
                 "do not build a holdout from previously used data")

    strata = defaultdict(list)
    for tid in eligible:
        _, pos, bucket, _ = pool[tid]
        strata[f"{pos}|{bucket}"].append(tid)
    chosen = allocate(strata, args.n, random.Random(SEED))
    ids = sorted(t for v in chosen.values() for t in v)

    final = {k: {**views["dev"][k], **views["test"][k]} for k in views["dev"]}
    keep = set(ids)
    final["targets"] = {t: v for t, v in final["targets"].items() if t in keep}
    final["substitutes"] = {s: v for s, v in final["substitutes"].items() if v["target_id"] in keep}
    final["substitute_labels"] = {s: final["substitute_labels"][s] for s in final["substitutes"]}
    final["contexts"] = {c: v for c, v in final["contexts"].items()
                         if c in {t["context_id"] for t in final["targets"].values()}}
    data_path = data_dir / "coinco_sealed.json.gz"
    write_json_gz(data_path, final)

    payload = {
        "ids": ids,
        "n": len(ids),
        "seed": SEED,
        "strata": {k: len(v) for k, v in chosen.items()},
        "stratum_pool_sizes": {k: len(v) for k, v in sorted(strata.items())},
        "frequency_buckets": dict(zip(FREQ_LABELS, ("<=3", "4-8", "9-21", ">=22"))),
        "split_of_id": {t: pool[t][0] for t in ids},
        "tokenizers": args.tokenizers.split(","),
        "pool_counts": dict(sorted(counts.items())),
        "eligible_all_tokenizers": len(eligible),
        "per_tokenizer": {k: {"eligible": v["eligible"], "excluded_n": len(v["excluded"]),
                              "excluded": v["excluded"]} for k, v in per_model.items()},
        "source_sha256": {p.name: C.sha256(p) for p in [RAW / "coinco_dev.json.gz",
                                                       RAW / "coinco_test.json.gz", *USED, *SWORDS]},
        "eval_file": str(data_path.relative_to(C.MAIN)) if data_path.is_relative_to(C.MAIN) else str(data_path),
        "eval_file_sha256": C.sha256(data_path),
        "substitutes": len(final["substitutes"]),
        "created_utc": C.now_utc(),
        "git_commit": C.git_commit(),
    }
    C.write_json(ids_path, payload)
    write_report(payload, out_dir)
    print(f"{len(ids)} sealed targets of {len(eligible)} eligible -> {ids_path}")


def write_report(p, out_dir):
    lines = ["# Sealed holdout: creation report", "",
             f"Created {p['created_utc']} at commit `{p['git_commit']}` by "
             "`scripts/round2/build_sealed_holdout.py`. No model was loaded or scored.", "",
             f"**{p['n']} targets** drawn (seed {p['seed']}) from **{p['eligible_all_tokenizers']}** "
             f"eligible CoInCo targets; {p['substitutes']} substitutes.", "",
             "## Pool construction and exclusions", "", "| step | targets |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in p["pool_counts"].items()]
    lines += ["", "## Tokenizer eligibility (evaluator's own skip rules, both scorers)", "",
              "| tokenizer | eligible | excluded |", "|---|---|---|"]
    lines += [f"| {k} | {v['eligible']} | {v['excluded_n']} |" for k, v in p["per_tokenizer"].items()]
    lines += ["", "Only targets eligible for all three tokenizers are sampled.", "",
              "## Strata (POS x lemma-frequency bucket)", "",
              "Frequency = count of the target's (lemma, POS) among all 15,415 CoInCo targets; "
              "bucket edges are that distribution's quartiles (<=3, 4-8, 9-21, >=22).", "",
              "| stratum | eligible | drawn |", "|---|---|---|"]
    lines += [f"| {k} | {p['stratum_pool_sizes'][k]} | {p['strata'].get(k, 0)} |"
              for k in p["stratum_pool_sizes"]]
    lines += ["", "## Pinned files", ""] + [f"- `{k}` {v}" for k, v in p["source_sha256"].items()]
    lines += [f"- evaluation file `{p['eval_file']}` {p['eval_file_sha256']} (derived, not committed)", ""]
    (out_dir / "holdout_creation_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
