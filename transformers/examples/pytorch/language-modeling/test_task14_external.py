import gzip
import ast
import csv
import json
import math
import os
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from compare_task14_results import paired_rows  # noqa: E402
from eval_swords import official_gap  # noqa: E402
from prepare_swords_concept_sets import build_rows  # noqa: E402
from verify_task14_data import sha256  # noqa: E402


def test_local_gap_matches_checksum_pinned_swords_reference_fixtures():
    human = [3.0, 2.0, 0.0]
    assert math.isclose(official_gap([3.0, 2.0, 1.0], human), 1.0)
    assert math.isclose(official_gap([1.0, 2.0, 3.0], human), 0.4848484848484849)
    assert math.isclose(official_gap([2.0, 3.0, 1.0], human), 0.8181818181818182)


def test_swords_dev_split_is_context_grouped_and_test_ids_are_disjoint():
    with gzip.open(os.path.join(ROOT, "data/swords/swords-v1.1_dev.json.gz"), "rt") as handle:
        dev = json.load(handle)
    with gzip.open(os.path.join(ROOT, "data/swords/swords-v1.1_test.json.gz"), "rt") as handle:
        test = json.load(handle)
    args = SimpleNamespace(
        split_seed=42, train_fraction=0.8, pos=None, max_context_chars=600,
        positive_threshold=0.5, negative_threshold=0.0, min_positives=2,
        max_negatives=10,
    )
    rows, _ = build_rows(dev, args)
    train_contexts = {row["context_id"] for row in rows if row["partition"] == "train"}
    val_contexts = {row["context_id"] for row in rows if row["partition"] == "val"}
    train_targets = {row["target_id"] for row in rows if row["partition"] == "train"}
    val_targets = {row["target_id"] for row in rows if row["partition"] == "val"}
    assert train_contexts.isdisjoint(val_contexts)
    assert train_targets.isdisjoint(val_targets)
    assert (train_contexts | val_contexts).isdisjoint(set(test["contexts"]))
    assert (train_targets | val_targets).isdisjoint(set(test["targets"]))


def test_paired_comparison_rejects_nonidentical_rows():
    try:
        paired_rows([{"target_id": "a"}], [{"target_id": "b"}], "target_id")
    except ValueError as error:
        assert "identical rows" in str(error)
    else:
        raise AssertionError("nonidentical paired rows were accepted")


def test_manifest_hashes_match_local_swords_and_bm_files():
    manifest_path = os.path.join(ROOT, "data/external_benchmarks_manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    entries = {entry["name"]: entry for entry in manifest["files"]}
    for name in ["swords_dev", "swords_test", "bm_semlex_curated_200"]:
        path = os.path.join(ROOT, entries[name]["path"])
        assert sha256(__import__("pathlib").Path(path)) == entries[name]["sha256"]


def test_inclusive_and_exclusive_arms_are_row_matched_and_differ_only_by_gold():
    roots = [
        "data/swords/concept_dev",
        "data/syn/youtube_clean_gold",
        "data/hyp/youtube_clean_gold",
    ]
    for root in roots:
        for split in ("train", "val"):
            inclusive_path = os.path.join(ROOT, root, f"context_loss_{split}.csv")
            exclusive_path = os.path.join(ROOT, root, f"context_loss_{split}_goldexcl.csv")
            with open(inclusive_path, encoding="utf-8", newline="") as handle:
                inclusive = list(csv.DictReader(handle))
            with open(exclusive_path, encoding="utf-8", newline="") as handle:
                exclusive = list(csv.DictReader(handle))
            assert [row["row_id"] for row in inclusive] == [row["row_id"] for row in exclusive]
            for inc, exc in zip(inclusive, exclusive):
                assert inc["text"] == exc["text"]
                assert inc["gold_surface"] == exc["gold_surface"]
                gold = inc["gold_surface"].casefold()
                inc_set = {str(value).casefold() for value in ast.literal_eval(inc["context_syn"])}
                exc_set = {str(value).casefold() for value in ast.literal_eval(exc["context_syn"])}
                assert gold in inc_set and gold not in exc_set
                assert inc_set == exc_set | {gold}


SWORDS_DIR = os.path.join(ROOT, "data/swords/concept_dev")


def _read_lines(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _cycle_to_size(rows, size):
    """The notebook's budget-matching rule, mirrored so the test pins the same behaviour."""
    return [rows[index % len(rows)] for index in range(size)]


def test_swords_augmentation_corpus_reproduces_iyer_data_augmentation():
    """One instance per gold-exclusive alternative, with the observed target substituted out.

    This is the arm whose size fixes the volume budget for every other arm, so a silent change
    here rescales the whole comparison.
    """
    for split in ("train", "val", "full"):
        with open(os.path.join(SWORDS_DIR, f"context_loss_{split}_goldexcl.csv"),
                  encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        augmented = _read_lines(os.path.join(SWORDS_DIR, f"context_syn_{split}.txt"))
        expected = [f'{row["text"]} {candidate}'
                    for row in rows for candidate in ast.literal_eval(row["context_syn"])]
        assert augmented == [line.strip() for line in expected]

        # The gold continuation lives in vanilla_{split}.txt; it must not be an augmented target.
        golds = {f'{row["text"]} {row["gold_surface"]}'.strip() for row in rows}
        assert not (set(augmented) & golds)

    report = json.load(open(os.path.join(SWORDS_DIR, "prepare_report.json"), encoding="utf-8"))
    assert report["augmentation_lines"]["train"] == len(
        _read_lines(os.path.join(SWORDS_DIR, "context_syn_train.txt"))
    )


def test_every_task14_arm_resolves_to_the_same_matched_volume(tmp_path):
    """All six arms must preprocess to exactly BUDGET rows.

    Guards the notebook's `--no-deduplicate_text_rows`: text-row dedup defaults ON, and would
    collapse A0's repeated originals and the hybrid arms' replay back to 251 unique lines,
    undoing the volume matching without raising anything.
    """
    import pytest

    pytest.importorskip("datasets")
    from run_clm_sequence_ncp import _read_train_rows  # noqa: E402

    budget = len(_read_lines(os.path.join(SWORDS_DIR, "context_syn_train.txt")))
    originals = _read_lines(os.path.join(SWORDS_DIR, "vanilla_train.txt"))
    with open(os.path.join(SWORDS_DIR, "context_loss_train_goldexcl.csv"),
              encoding="utf-8", newline="") as handle:
        concept = list(csv.DictReader(handle))

    a0 = tmp_path / "a0.txt"
    a0.write_text("\n".join(_cycle_to_size(originals, budget)) + "\n", encoding="utf-8")
    replay = tmp_path / "replay.txt"
    replay.write_text(
        "\n".join(_cycle_to_size(originals, budget - len(concept))) + "\n", encoding="utf-8"
    )
    p1 = tmp_path / "p1.csv"
    with open(p1, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(concept[0]))
        writer.writeheader()
        writer.writerows(_cycle_to_size(concept, budget))

    def rows_for(path):
        return len(_read_train_rows(str(path), "context_syn", "train",
                                    deduplicate_text_rows=False))

    # Paper arms train on one file; hybrid arms are concept rows plus replay padding.
    assert rows_for(a0) == budget
    assert rows_for(os.path.join(SWORDS_DIR, "context_syn_train.txt")) == budget
    assert rows_for(p1) == budget
    assert len(concept) + rows_for(replay) == budget

    # The tripwire: leaving dedup on silently shrinks the repeated-original arms.
    deduped = len(_read_train_rows(str(a0), "context_syn", "train",
                                   deduplicate_text_rows=True))
    assert deduped == len(originals) < budget


def test_rejected_mass_share_is_the_suppression_metric_auroc_cannot_provide():
    """AUROC measures RANKING of rejected below acceptable; this measures actual probability.

    The two come apart exactly where the contrastive claim lives: a model can rank every
    acceptable substitute above every rejected one (AUROC = 1.0) while still placing most of the
    candidate set's mass on the rejected ones.  Regression test for the direction convention too,
    since ``compare_task14_results`` reads a negative delta as an improvement.
    """
    import torch

    def share(alt_logps, rejected_logps):
        """Mirror of the computation in eval_swords.py."""
        alt_nll = float(-torch.logsumexp(torch.tensor(alt_logps, dtype=torch.float64), dim=0))
        rej_nll = float(-torch.logsumexp(torch.tensor(rejected_logps, dtype=torch.float64), dim=0))
        return float(torch.sigmoid(torch.tensor(alt_nll - rej_nll, dtype=torch.float64)))

    # Equal mass on each side -> exactly half.
    assert share([math.log(0.1)], [math.log(0.1)]) == pytest.approx(0.5)

    # Acceptable strictly preferred -> share below a half; suppressed further -> smaller still.
    confident = share([math.log(0.30)], [math.log(0.01)])
    lukewarm = share([math.log(0.30)], [math.log(0.20)])
    assert confident < lukewarm < 0.5
    assert confident == pytest.approx(0.01 / 0.31)          # closed form
    assert lukewarm == pytest.approx(0.20 / 0.50)

    # It sums over the whole rejected set, not just the best one.
    many = share([math.log(0.30)], [math.log(0.05)] * 4)
    one = share([math.log(0.30)], [math.log(0.05)])
    assert many > one, "mass on rejected substitutes must accumulate across them"

    # The case AUROC cannot see: perfect ranking, terrible mass allocation.  Every acceptable
    # candidate outscores every rejected one, yet rejected substitutes hold most of the mass.
    acceptable = [math.log(0.02)]
    rejected = [math.log(0.019)] * 40
    from eval_swords import auroc
    scores = acceptable + rejected
    labels = [True] + [False] * len(rejected)
    assert auroc(scores, labels) == pytest.approx(1.0)      # ranking is perfect
    assert share(acceptable, rejected) > 0.9                # mass allocation is not
