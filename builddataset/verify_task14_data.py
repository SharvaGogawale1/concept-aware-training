#!/usr/bin/env python3
"""Fetch missing Task-14 benchmarks from pinned commits and verify their integrity/schema."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath([str(root), str(target)]) != str(root):
                raise ValueError(f"unsafe archive member: {member.filename}")
        handle.extractall(destination)


def verify_swords(path: Path, expected_targets: int) -> Dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    required = {"contexts", "targets", "substitutes", "substitute_labels"}
    if not required <= set(data):
        raise ValueError(f"{path}: missing keys {sorted(required - set(data))}")
    if len(data["targets"]) != expected_targets or len(data["contexts"]) != expected_targets:
        raise ValueError(f"{path}: unexpected context/target counts")
    offset_errors = 0
    for target in data["targets"].values():
        context = data["contexts"][target["context_id"]]["context"]
        offset = int(target["offset"])
        offset_errors += context[offset:offset + len(target["target"])] != target["target"]
    if offset_errors:
        raise ValueError(f"{path}: {offset_errors} target offsets do not align")
    return {"contexts": len(data["contexts"]), "targets": len(data["targets"]),
            "substitutes": len(data["substitutes"]), "offset_errors": offset_errors}


def verify_hyperlex(path: Path) -> Dict[str, int]:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().split()
        rows = [line.split() for line in handle if line.strip()]
    if header[:5] != ["WORD1", "WORD2", "POS", "TYPE", "AVG_SCORE"]:
        raise ValueError(f"{path}: unexpected header")
    return {"pairs": len(rows)}


def verify_bm(path: Path) -> Dict[str, int]:
    usable = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if len(fields) < 5:
                raise ValueError(f"{path}: malformed row {usable}")
            target, _, _, index_text, sentence = fields[:5]
            tokens = sentence.split()
            index = int(index_text)
            normalized = tokens[index].strip(".,;:!?\"'").casefold()
            if normalized != target.casefold():
                raise ValueError(f"{path}: target/index mismatch at row {usable}")
            usable += 1
    return {"rows": usable}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--manifest", default="data/external_benchmarks_manifest.json")
    parser.add_argument("--download_missing", action="store_true")
    parser.add_argument("--report_json", default=None)
    return parser.parse_args()


def verify_gap_reference(path: Path) -> Dict[str, Any]:
    """Execute fixed fixtures with the checksum-pinned GAP implementation used by SWORDS."""
    spec = importlib.util.spec_from_file_location("task14_gap_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import GAP reference {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gap = module.GeneralizedAveragePrecision
    gold = [["a", 3.0], ["b", 2.0], ["c", 0.0]]
    fixtures = {
        "perfect": ([["a", 3.0], ["b", 2.0], ["c", 1.0]], 1.0),
        "reverse": ([["a", 1.0], ["b", 2.0], ["c", 3.0]], 0.4848484848484849),
        "swap_top": ([["a", 2.0], ["b", 3.0], ["c", 1.0]], 0.8181818181818182),
    }
    observed = {name: float(gap.calc(gold, predicted))
                for name, (predicted, _) in fixtures.items()}
    for name, (_, expected) in fixtures.items():
        if abs(observed[name] - expected) > 1e-12:
            raise ValueError(
                f"official GAP fixture {name} changed: {observed[name]} != {expected}"
            )
    return {"sha256": sha256(path), "fixtures": observed}


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    manifest_path = root / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report: Dict[str, Any] = {
        "manifest": str(manifest_path), "files": {}, "archives": {},
        "metric_references": {},
    }

    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.exists() and args.download_missing:
            fetch(entry["url"], path)
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; rerun with --download_missing")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {path}: {actual}")
        report["files"][entry["name"]] = {"path": str(path), "sha256": actual}

    for entry in manifest["archives"]:
        target_dir = root / entry["target_dir"]
        missing = [relative for relative in entry["required_files"] if not (target_dir / relative).exists()]
        if missing and args.download_missing:
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "benchmark.zip"
                fetch(entry["url"], archive)
                if sha256(archive) != entry["sha256"]:
                    raise ValueError(f"archive checksum mismatch for {entry['name']}")
                safe_extract(archive, target_dir)
        verified = {}
        for relative, expected in entry["required_files"].items():
            path = target_dir / relative
            if not path.exists():
                raise FileNotFoundError(f"missing {path}; rerun with --download_missing")
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"checksum mismatch for {path}: {actual}")
            verified[relative] = actual
        report["archives"][entry["name"]] = verified

    # Metric code is not vendored. During the reproducible Colab setup, fetch the exact pinned
    # reference, verify its digest, execute fixtures, and discard it with the temporary directory.
    for entry in manifest.get("metric_references", []):
        if not args.download_missing:
            report["metric_references"][entry["name"]] = {
                "status": "not_fetched_without_--download_missing",
                "expected_sha256": entry["sha256"],
            }
            continue
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "metric_reference.py"
            fetch(entry["url"], reference)
            actual = sha256(reference)
            if actual != entry["sha256"]:
                raise ValueError(
                    f"metric-reference checksum mismatch for {entry['name']}: {actual}"
                )
            report["metric_references"][entry["name"]] = verify_gap_reference(reference)

    report["schema"] = {
        "swords_dev": verify_swords(root / "data/swords/swords-v1.1_dev.json.gz", 370),
        "swords_test": verify_swords(root / "data/swords/swords-v1.1_test.json.gz", 762),
        "hyperlex_all": verify_hyperlex(root / "data/hyperlex-data/hyperlex-all.txt"),
        "bm_semlex": verify_bm(root / "data/bm_semlex/curated_200.tsv"),
    }
    if args.report_json:
        output = Path(args.report_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
