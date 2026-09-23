#!/usr/bin/env python3
"""Two more human-labelled substitution benchmarks, in the SWORDS JSON layout.

SWORDS is one dataset.  A substitution gain that shows up on it alone is one
result; the same gain on the two benchmarks it was built to succeed is a
pattern.  Both are candidate-ranking tasks, so `eval_swords.py` scores them
unchanged once they are in its input format:

  contexts[c]           {"context": str}
  targets[t]            {"context_id", "target", "offset", "pos"}
  substitutes[s]        {"target_id", "substitute"}
  substitute_labels[s]  ["TRUE"] * annotator_count  or  ["FALSE"]

Human weight is the annotator count, encoded as that many TRUE labels: the
evaluator's `human_true_count` is then the gold frequency GAP was defined
with (Thater et al., 2010), and its `human` score is 1 for accepted, 0 for
rejected.  No UNSURE labels exist in either source.

SemEval-2007 Task 10 (McCarthy & Navigli): every test instance whose gold
line is non-empty (1,703 of 1,710).  The candidate pool of an instance is the
union of gold substitutes of its lexelt over the whole dataset -- Melamud et
al.'s `lst.gold.candidates`, the standard ranking setting -- plus its own gold.

CoInCo (Kremer et al., 2014): the parsed files the SWORDS repository ships at
the commit this repo pins, whose implicit labels (TRUE_IMPLICIT = provided by
an annotator, FALSE_IMPLICIT = in the lexelt's pool but not provided) become
TRUE x freq / FALSE.  Scoring all 5,388 test targets costs ~10x SWORDS dev per
checkpoint, so a FIXED subset of 800 targets is drawn once, with seed 0, from
the targets that have both an accepted and a rejected substitute.  The draw
happens here, before any model is scored, and the output is byte-stable
(sorted keys, gzip mtime 0) so the committed file and a rebuild agree.

Raw inputs are pinned (URL + SHA-256) in data/external_benchmarks_manifest.json
and fetched by builddataset/verify_task14_data.py --download_missing.

    python builddataset/build_lexsub_benchmarks.py [--repo_root .] [--force]
"""
import argparse
import gzip
import hashlib
import html
import io
import json
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path

RAW = Path("data/lexsub/raw")
OUT = Path("data/lexsub")
COINCO_SUBSET = 800
COINCO_SEED = 0
POS = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}
# The derived files are not committed (no benchmark data in this repo is), so their
# digests are pinned here instead: a rebuild that differs is an error, never a
# silently different benchmark.  Update these only together with a deliberate
# change to this script, and say so in the commit.
EXPECTED_SHA256 = {
    "semeval07_test.json.gz": "94872b2ceb7e6e0643708045ff3f5e799bd41254f2de9a77fc56e445645c3373",
    "coinco_dev_sub800.json.gz": "6679c2f22c1ee92469e6e24438d3b65edb86e1276ee19731af05d0d8f130962f",
    "coinco_test_sub800.json.gz": "4400bf20e4ede042446743c08e536d3d4479a38515d92e3331e2ab34745c2bf2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_gz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    # mtime=0: the gzip header carries a timestamp, and a rebuilt file must be
    # byte-identical to the committed one or the checksum check below is noise.
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    path.write_bytes(buffer.getvalue())


def _ids(prefix: str, *parts: str) -> str:
    return f"{prefix}:" + hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ SemEval-2007


def build_semeval07(tarball: Path, candidates_file: Path) -> dict:
    with tarfile.open(tarball) as archive:
        xml = archive.extractfile("test/lexsub_test.xml").read().decode("utf-8", "replace")
        gold_text = archive.extractfile("scoring/gold").read().decode("utf-8", "replace")

    pool = {}
    for line in candidates_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "::" not in line:
            continue
        lexelt, rest = line.split("::", 1)
        pool[lexelt.strip()] = [c.strip() for c in rest.split(";") if c.strip()]

    gold = defaultdict(dict)                      # (lexelt, instance) -> {substitute: freq}
    for line in gold_text.splitlines():
        if "::" not in line:
            continue
        head, rest = line.split("::", 1)
        lexelt, instance = head.split()
        for item in rest.strip().split(";"):
            item = item.strip()
            if not item:
                continue
            word, freq = item.rsplit(" ", 1)
            # A gold line can list one substitute twice (annotators' spellings that
            # lemmatise together); the counts add, they do not overwrite.
            gold[(lexelt, instance)][word.strip()] = gold[(lexelt, instance)].get(word.strip(), 0) + int(freq)

    data = {"contexts": {}, "targets": {}, "substitutes": {}, "substitute_labels": {}}
    # The XML is not well-formed (bare "&#8211 ;" entities), so it is read with
    # regexes and unescaped leniently, the way the task's own scorer does.
    for lexelt, body in re.findall(r'<lexelt item="([^"]+)">(.*?)</lexelt>', xml, re.S):
        pos = POS.get(lexelt.rsplit(".", 1)[-1], "?")
        for instance, context in re.findall(r'<instance id="(\d+)">\s*<context>(.*?)</context>', body, re.S):
            weights = gold.get((lexelt, instance))
            if not weights:
                continue                            # instances 1,704-1,710: no gold line
            left, rest = context.split("<head>", 1)
            head, right = rest.split("</head>", 1)
            left, head, right = (html.unescape(x).strip("\n") for x in (left, head, right))
            text = f"{left}{head}{right}"
            context_id = _ids("c", "semeval07", lexelt, instance)
            target_id = _ids("t", "semeval07", lexelt, instance)
            data["contexts"][context_id] = {"context": text}
            data["targets"][target_id] = {"context_id": context_id, "target": head,
                                          "offset": len(left), "pos": pos,
                                          "extra": {"lexelt": lexelt, "instance": instance}}
            candidates = list(dict.fromkeys(pool.get(lexelt, []) + list(weights)))
            for candidate in candidates:
                substitute_id = _ids("s", "semeval07", lexelt, instance, candidate)
                data["substitutes"][substitute_id] = {"target_id": target_id, "substitute": candidate}
                freq = weights.get(candidate, 0)
                data["substitute_labels"][substitute_id] = ["TRUE"] * freq if freq else ["FALSE"]
    return data


# ------------------------------------------------------------------------ CoInCo


def build_coinco(parsed: Path, subset: int, seed: int) -> dict:
    with gzip.open(parsed, "rt", encoding="utf-8") as handle:
        source = json.load(handle)

    by_target = defaultdict(list)
    for substitute_id, substitute in source["substitutes"].items():
        labels = source["substitute_labels"].get(substitute_id) or []
        if not labels:
            continue
        accepted = all(label.startswith("TRUE") for label in labels)
        rejected = all(label.startswith("FALSE") for label in labels)
        if not (accepted or rejected):
            raise ValueError(f"{parsed}: mixed labels on one substitute: {labels}")
        freq = 1
        for extra in substitute.get("extra", []):
            if not isinstance(extra, dict):
                continue
            attrs = (extra.get("coinco") or {}).get("xml_attrs") or {}
            if attrs.get("freq"):
                freq = int(attrs["freq"])
        by_target[substitute["target_id"]].append(
            (substitute_id, substitute["substitute"], freq if accepted else 0))

    eligible = []
    for target_id in sorted(source["targets"]):
        subs = by_target.get(target_id, [])
        if any(f > 0 for _, _, f in subs) and any(f == 0 for _, _, f in subs):
            eligible.append(target_id)
    chosen = sorted(random.Random(seed).sample(eligible, subset)) if subset else eligible

    data = {"contexts": {}, "targets": {}, "substitutes": {}, "substitute_labels": {}}
    for target_id in chosen:
        target = source["targets"][target_id]
        context_id = target["context_id"]
        data["contexts"][context_id] = {"context": source["contexts"][context_id]["context"]}
        data["targets"][target_id] = {"context_id": context_id, "target": target["target"],
                                      "offset": int(target["offset"]), "pos": target.get("pos", "?")}
        for substitute_id, substitute, freq in by_target[target_id]:
            data["substitutes"][substitute_id] = {"target_id": target_id, "substitute": substitute}
            data["substitute_labels"][substitute_id] = ["TRUE"] * freq if freq else ["FALSE"]
    data["subset"] = {"n": len(chosen), "of_eligible": len(eligible),
                      "of_all": len(source["targets"]), "seed": seed}
    return data


def summary(data: dict) -> str:
    labels = data["substitute_labels"]
    accepted = sum(1 for v in labels.values() if v and v[0] == "TRUE")
    return (f"{len(data['targets'])} targets, {len(labels)} substitutes, "
            f"{accepted} accepted ({accepted / max(len(labels), 1):.1%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--force", action="store_true", help="rebuild even if the outputs exist")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    raw, out = root / RAW, root / OUT

    builds = {
        "semeval07_test.json.gz": lambda: build_semeval07(raw / "task10data.tar.gz", raw / "lst.gold.candidates"),
        f"coinco_dev_sub{COINCO_SUBSET}.json.gz": lambda: build_coinco(raw / "coinco_dev.json.gz", COINCO_SUBSET, COINCO_SEED),
        f"coinco_test_sub{COINCO_SUBSET}.json.gz": lambda: build_coinco(raw / "coinco_test.json.gz", COINCO_SUBSET, COINCO_SEED),
    }
    for name, build in builds.items():
        path = out / name
        if path.is_file() and not args.force:
            print(f"exists, kept: {path.relative_to(root)}  sha256 {sha256(path)[:16]}")
            continue
        data = build()
        write_json_gz(path, data)
        digest = sha256(path)
        print(f"wrote {path.relative_to(root)}: {summary(data)}  sha256 {digest[:16]}")
        expected = EXPECTED_SHA256.get(name)
        if expected and digest != expected:
            raise SystemExit(f"{name}: rebuilt digest {digest} != pinned {expected}; "
                             "the benchmark changed -- stop and find out why")


if __name__ == "__main__":
    main()
