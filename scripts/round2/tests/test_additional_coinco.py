"""Synthetic tests for the additional CoInCo evaluation (runner and analysis).  No model, no GPU.

    python -m pytest scripts/round2/tests -q
"""
import gzip
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

import additional_coinco_eval as R
import analyze_additional_coinco as Z
import analyze_round2 as A
import evaluate_round2 as E
import r2common as C

MODELS = tuple(C.FAMILIES)
REPS = 300
# GAP effect of each configuration over Concept loss (raw scale; x100 = points)
EFFECT = {"concept": 0.0, "alt_mass": 0.02, "alt_balance": 0.0, "alt_mass_balance": 0.02, A.WHOLESET: 0.004}


# ---------- synthetic data ----------

def make_eval(tmp, n_sentences=60, seed=0):
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    targets, sentence_of = {}, {}
    for s in range(n_sentences):
        for k in range(1 + s % 4):
            t = f"t:{s:03d}_{k}"
            targets[t] = {"context_id": f"c:{s:03d}", "target": "w", "offset": 0, "pos": "NOUN"}
            sentence_of[t] = f"c:{s:03d}"
    eval_file = tmp / "eval.json.gz"
    with gzip.open(eval_file, "wt", encoding="utf-8") as handle:
        json.dump({"targets": targets, "contexts": {}, "substitutes": {}, "substitute_labels": {}}, handle)
    ids_file = tmp / "ids.json"
    ids_file.write_text(json.dumps({"ids": sorted(targets), "eval_file_sha256": C.sha256(eval_file)}))
    base = {t: rng.uniform(0.3, 0.6) for t in targets}
    sent = {c: rng.normal(0, 0.01) for c in set(sentence_of.values())}
    return eval_file, ids_file, base, sent, sentence_of


def output_record(ckpt, targets, gap_of, drop=()):
    rec = {"checkpoint": ckpt}
    for mode in R.MODES:
        per = []
        for t in targets:
            if t in drop:
                continue
            gold = 5.0 + 0.1 * len(t)
            per.append({"target_id": t, "gap": gap_of(t, mode), "alternatives_nll": 7.0 - gap_of(t, mode),
                        "gold_nll": gold, "inclusive_nll": gold - 0.3 - gap_of(t, mode)})
        rec[mode] = {"per_target": per, "gap": float(np.mean([p["gap"] for p in per])),
                     "alternatives_nll": float(np.mean([p["alternatives_nll"] for p in per]))}
    return [rec]


def ckpt_of(tag, name):
    return f"/ckpt/{tag}/{name}"


def build_root(tmp, missing=None, n_sentences=60):
    """A scoring root with all three models; missing: {tag: [flat, ...]} recorded as missing runs."""
    missing = missing or {}
    eval_file, ids_file, base, sent, sentence_of = make_eval(tmp, n_sentences)
    rng = np.random.default_rng(1)
    root = tmp / "root"
    for tag in MODELS:
        outputs, miss = {}, {}
        for objective, gamma, seed, source in R.plan_arms():
            name = A.flat(objective, gamma, seed)
            ckpt = ckpt_of(tag, name)
            if name in missing.get(tag, ()):
                miss[name] = {"checkpoint": ckpt, "source": source, "attempts": 3, "exit_codes": [1, 1, 1]}
                continue
            noise = {t: rng.normal(0, 0.005) for t in base}
            shift = 0.003 * C.SEEDS.index(seed)

            def gap_of(t, mode, objective=objective, noise=noise, shift=shift):
                return base[t] + sent[sentence_of[t]] + EFFECT[objective] + noise[t] + shift + (0.001 if mode == "full" else 0)

            f = root / tag / "outputs" / f"{name}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(output_record(ckpt, sorted(base), gap_of)))
            outputs[name] = {"checkpoint": ckpt, "source": source, "sha256": C.sha256(f), "attempts": 1}
        (root / tag / "COMPLETE.json").write_text(json.dumps(
            {"model": tag, "plan_sha256": "p", "pins": {}, "correction": None, "outputs": outputs, "missing": miss}))
    return root, eval_file, ids_file, sentence_of


def analyse(tmp, **kw):
    root, eval_file, ids_file, sentence_of = build_root(tmp, **kw)
    reading, rows = Z.run(root, eval_file, ids_file, tmp / "analysis", REPS, tmp / "no_effects.csv")
    return reading, rows, root, eval_file, ids_file, sentence_of


def row(rows, cid, tag, scorer="left", metric="gap"):
    (r,) = [r for r in rows if r["id"] == cid and r["model"] == tag and r["scorer"] == scorer and r["metric"] == metric]
    return r


# ---------- analysis: verdicts and the safeguard ----------

def base_row(**kw):
    r = {"id": "D1", "model": "m", "scorer": "left", "metric": "gap", "computed": True, "seeds": tuple(C.SEEDS),
         "n_items": 100, "n_expected": 100, "estimate": 0.01, "p_holm": 0.001}
    r.update(kw)
    return r


def test_verdict_rules():
    assert Z.verdict(base_row()) == Z.CONFIRMED
    assert Z.verdict(base_row(estimate=-0.01)) == Z.ADVERSE
    assert Z.verdict(base_row(p_holm=0.2)) == Z.INCONCLUSIVE
    assert Z.verdict(base_row(p_holm=0.2, estimate=-0.01)) == Z.INCONCLUSIVE      # negative, spans zero: not a failure
    assert Z.verdict(base_row(metric="alternatives_nll")) == Z.DESCRIPTIVE
    assert Z.verdict(base_row(computed=False)) == Z.NOT_COMPUTED


@pytest.mark.parametrize("change", [{"seeds": (42, 123)}, {"seeds": (42,)}, {"seeds": (123, 2024)},
                                    {"n_items": 99}, {"seeds": (42, 123), "metric": "alternatives_nll"}])
def test_incomplete_comparison_never_gets_a_verdict(change):
    """The safeguard: a decisive p and a large effect still give 'incomplete'."""
    assert Z.verdict(base_row(p_holm=1e-6, estimate=0.5, **change)) == Z.INCOMPLETE
    assert Z.verdict(base_row(p_holm=1e-6, estimate=-0.5, **change)) == Z.INCOMPLETE


@pytest.mark.parametrize("forged", [{"seeds": (42, 123)}, {"n_items": 99}, {"computed": False},
                                    {"metric": "observed_share"}, {"p_holm": 0.3}])
def test_safeguard_rejects_a_forged_confirmatory_row(forged):
    with pytest.raises(RuntimeError, match="safeguard"):
        Z.check_safeguard([base_row(verdict=Z.CONFIRMED, **forged)])
    with pytest.raises(RuntimeError, match="safeguard"):
        Z.check_safeguard([base_row(verdict=Z.ADVERSE, estimate=-0.01, **forged)])


def test_complete_evaluation(tmp_path):
    reading, rows, root, eval_file, ids_file, sentence_of = analyse(tmp_path)
    n_sentences, n_targets = len(set(sentence_of.values())), len(sentence_of)
    for tag in MODELS:
        d1 = row(rows, "D1", tag)
        assert d1["verdict"] == Z.CONFIRMED and d1["seeds"] == tuple(C.SEEDS) and d1["holm_m"] == 3
        assert d1["n_sentences"] == n_sentences and d1["n_items"] == n_targets
        assert d1["estimate"] == pytest.approx(EFFECT["alt_mass"], abs=0.002)
        assert row(rows, "D3", tag)["estimate"] == pytest.approx(0.0, abs=0.002)
        assert row(rows, "D3", tag)["verdict"] == Z.INCONCLUSIVE
        assert row(rows, "D1", tag, metric="alternatives_nll")["verdict"] == Z.DESCRIPTIVE
    assert reading["D1"]["summary"] == "confirmed in all three" and reading["D1"]["general_claim"]
    assert reading["D3"]["confirmed"] == [] and not reading["D3"]["general_claim"]
    table = (tmp_path / "analysis" / "table.md").read_text()
    lines = {line.split(" | ")[0].strip("| "): line for line in table.splitlines() if line.startswith("|")}
    assert lines["Alt-Mass − Concept loss"].count("†") == 3 and "†" not in lines["Alt-Mass+Balance − Alt-Mass"]
    assert "‡" not in table and "Missing runs" not in table
    for f in ("cells.csv", "comparisons.csv", "readings.json", "table.md", "report.md"):
        assert (tmp_path / "analysis" / f).is_file()


def test_estimate_is_the_target_mean_difference(tmp_path):
    """D1 equals the plain mean over targets and seeds of Alt-Mass minus Concept loss."""
    reading, rows, root, *_ = analyse(tmp_path)
    tag = MODELS[0]
    diffs = []
    for seed in C.SEEDS:
        a = json.loads((root / tag / "outputs" / f"{A.flat('alt_mass', C.PRIMARY_GAMMA, seed)}.json").read_text())[0]
        b = json.loads((root / tag / "outputs" / f"{A.flat('concept', None, seed)}.json").read_text())[0]
        diffs.append(np.mean([x["gap"] - y["gap"] for x, y in zip(a["left"]["per_target"], b["left"]["per_target"])]))
    assert row(rows, "D1", tag)["estimate"] == pytest.approx(float(np.mean(diffs)), abs=1e-12)


def test_missing_alt_mass_seed_blocks_every_verdict_that_uses_it(tmp_path):
    tag = MODELS[1]
    gone = A.flat("alt_mass", C.PRIMARY_GAMMA, 2024)
    reading, rows, *_ = analyse(tmp_path, missing={tag: [gone]})
    for cid in Z.COMPARISONS:                       # every comparison involves Alt-Mass
        r = row(rows, cid, tag)
        assert r["seeds"] == (42, 123) and r["verdict"] == Z.INCOMPLETE and r["holm_m"] == 3
    assert row(rows, "D1", tag)["estimate"] == pytest.approx(EFFECT["alt_mass"], abs=0.003)   # still reported
    assert row(rows, "D1", tag)["p_holm"] < Z.ALPHA                                          # decisive, yet no verdict
    for other in (MODELS[0], MODELS[2]):
        assert row(rows, "D1", other)["verdict"] == Z.CONFIRMED
    assert reading["D1"]["summary"] == "confirmed in 2 of 3" and reading["D1"]["incomplete"] == [tag]
    assert not reading["D1"]["general_claim"]
    table = (tmp_path / "analysis" / "table.md").read_text()
    assert "(2 seeds)" in table and "‡" in table and f"{tag} {gone}" in table


def test_missing_balance_seed_leaves_direct_comparisons_complete(tmp_path):
    tag = MODELS[2]
    reading, rows, *_ = analyse(tmp_path, missing={tag: [A.flat("alt_balance", C.PRIMARY_GAMMA, 42)]})
    for cid in ("D1", "D2", "D3"):
        assert row(rows, cid, tag)["seeds"] == tuple(C.SEEDS)
    assert row(rows, "D1", tag)["verdict"] == Z.CONFIRMED
    for cid in ("M1", "M2"):
        r = row(rows, cid, tag)
        assert r["seeds"] == (123, 2024) and r["verdict"] == Z.INCOMPLETE
    assert reading["D1"]["general_claim"] and reading["M1"]["incomplete"] == [tag]


def test_comparison_without_any_matched_seed(tmp_path):
    tag = MODELS[0]
    gone = [A.flat(A.WHOLESET, C.PRIMARY_GAMMA, s) for s in C.SEEDS]
    reading, rows, *_ = analyse(tmp_path, missing={tag: gone})
    r = row(rows, "D2", tag)
    assert not r["computed"] and r["verdict"] == Z.NOT_COMPUTED and r["p_holm"] is None
    assert reading["D2"]["incomplete"] == [tag] and not reading["D2"]["general_claim"]
    assert all(row(rows, "D2", m)["holm_m"] == 3 for m in MODELS)


def test_holm_family_size_stays_three(tmp_path):
    """A missing comparison enters Holm with p = 1, so the others are not corrected over fewer tests."""
    tag = MODELS[0]
    _, rows, *_ = analyse(tmp_path, missing={tag: [A.flat(A.WHOLESET, C.PRIMARY_GAMMA, s) for s in C.SEEDS]})
    ps = [row(rows, "D2", m)["p"] for m in MODELS[1:]]
    assert sorted(row(rows, "D2", m)["p_holm"] for m in MODELS[1:]) == pytest.approx(
        sorted(A.holm([1.0] + ps)[1:]))


def test_bootstrap_is_deterministic(tmp_path):
    _, rows1, *_ = analyse(tmp_path / "a")
    _, rows2, *_ = analyse(tmp_path / "b")
    key = lambda r: (r["id"], r["model"], r["scorer"], r["metric"])
    assert [(key(r), r["ci_low"], r["ci_high"], r["p"]) for r in sorted(rows1, key=key)] == \
           [(key(r), r["ci_low"], r["ci_high"], r["p"]) for r in sorted(rows2, key=key)]


def test_analysis_refuses_overwrite_tampering_and_gaps(tmp_path):
    root, eval_file, ids_file, _ = build_root(tmp_path)
    Z.run(root, eval_file, ids_file, tmp_path / "out", REPS, tmp_path / "none.csv")
    with pytest.raises(SystemExit, match="overwrite"):
        Z.run(root, eval_file, ids_file, tmp_path / "out", REPS, tmp_path / "none.csv")
    f = root / MODELS[0] / "outputs" / f"{A.flat('concept', None, 42)}.json"
    f.write_text(f.read_text().replace("0.", "0.0", 1))
    with pytest.raises(SystemExit, match="SHA-256"):
        Z.run(root, eval_file, ids_file, tmp_path / "out2", REPS, tmp_path / "none.csv")


def test_analysis_refuses_record_that_drops_a_run(tmp_path):
    root, eval_file, ids_file, _ = build_root(tmp_path)
    rec_file = root / MODELS[1] / "COMPLETE.json"
    rec = json.loads(rec_file.read_text())
    rec["outputs"].pop(A.flat("alt_mass", C.PRIMARY_GAMMA, 123))        # neither scored nor recorded missing
    rec_file.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="15 checkpoints"):
        Z.run(root, eval_file, ids_file, tmp_path / "out", REPS, tmp_path / "none.csv")


def test_analysis_refuses_output_missing_a_target(tmp_path):
    root, eval_file, ids_file, _ = build_root(tmp_path)
    tag, name = MODELS[2], A.flat("alt_balance", C.PRIMARY_GAMMA, 42)
    f = root / tag / "outputs" / f"{name}.json"
    rows = json.loads(f.read_text())
    rows[0]["left"]["per_target"].pop()
    f.write_text(json.dumps(rows))
    rec_file = root / tag / "COMPLETE.json"
    rec = json.loads(rec_file.read_text())
    rec["outputs"][name]["sha256"] = C.sha256(f)
    rec_file.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="exactly"):
        Z.run(root, eval_file, ids_file, tmp_path / "out", REPS, tmp_path / "none.csv")


# ---------- runner: gates ----------

PLAN_TEMPLATE = """# plan

## 11. Approval

- [{a}] Sharva Gogawale, date:{da}
- Chen Shani: not requested; not required for execution.
"""


def plan_text(a=" ", da=""):
    return PLAN_TEMPLATE.format(a=a, da=da)


def test_approvals_need_a_tick_and_a_date():
    assert R.approvals(plan_text()) == {"Sharva Gogawale": None}
    assert R.approvals(plan_text("x", " 2026-10-22")) == {"Sharva Gogawale": "2026-10-22"}
    assert R.approvals(plan_text("X", " 2026-10-22, approved revision 4")) == {"Sharva Gogawale": "2026-10-22"}
    assert R.approvals(plan_text("x", "")) == {"Sharva Gogawale": None}
    assert R.approvals(plan_text(" ", " 2026-10-22")) == {"Sharva Gogawale": None}
    assert R.approvals("no approval section") == {"Sharva Gogawale": None}


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_plan_gate_follows_commit_approval_and_pinned_hash(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    plan, protocol = repo / "plan.md", repo / "protocol.md"
    ok = lambda: []
    plan.write_text(plan_text())
    protocol.write_text("protocol\n")
    check = lambda: R.plan_problems(plan, protocol, repo, ok)
    assert "the plan is not committed" in check()
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "draft")
    problems = check()
    assert any("Sharva Gogawale" in p for p in problems)
    assert not any("Chen Shani" in p for p in problems)      # not requested, not required (revision 4)
    assert any("does not record the plan's SHA-256" in p for p in problems)
    protocol.write_text(f"Analysis plan SHA-256: {C.sha256(plan)}\n")
    git(repo, "commit", "-qam", "pinned, not approved")
    assert check() == ["the plan's approval line for Sharva Gogawale is not ticked and dated"]
    plan.write_text(plan_text("x", " 2026-10-22"))
    protocol.write_text(f"- Analysis plan SHA-256: `{C.sha256(plan)}`\n")
    git(repo, "commit", "-qam", "approved")
    assert check() == []
    plan.write_text(plan.read_text() + "\nedit\n")
    assert check() == ["the plan has uncommitted edits",
                       "the plan differs from the version whose SHA-256 ROUND2_PROTOCOL.md records"]
    git(repo, "commit", "-qam", "edit after approval")
    assert check() == ["the plan differs from the version whose SHA-256 ROUND2_PROTOCOL.md records"]
    assert R.plan_problems(plan, protocol, repo, lambda: ["not approved"])[-1] == "ROUND2_PROTOCOL.md not approved"


def test_real_plan_approval_lines():
    """The required approval line is in the form the gate reads.  Chen Shani's line records that her
    approval was not requested, and is never a ticked approval."""
    if not R.PLAN.is_file():
        pytest.skip("no plan in this checkout")
    body = R.PLAN.read_text().split("## 11. Approval", 1)[1]
    for name in R.APPROVERS:
        assert re.search(rf"^- \[[ xX]\] {re.escape(name)}, date:", body, re.M), name
    assert re.search(r"^- Chen Shani: not requested", body, re.M)
    assert not re.search(r"^- \[[xX]\] Chen Shani", body, re.M)


def test_preflight_needs_no_plan_approval_but_score_does(monkeypatch):
    monkeypatch.setattr(R, "plan_problems", lambda: ["the plan is not committed"])
    monkeypatch.setattr(R, "data_problems", lambda: [])
    monkeypatch.setattr(R, "resolve_checkpoints", lambda tag: ([], []))
    monkeypatch.setattr(R, "OUT_ROOT", Path("/nonexistent/additional_coinco"))
    ckpts, plan_gate, other = R.gates(MODELS[0])
    assert plan_gate == ["the plan is not committed"] and other == []


def test_eta():
    assert R.eta_hours([], 5) is None
    assert R.eta_hours([30.0, 30.0], 10) == pytest.approx(5.0)


def test_data_pins(tmp_path):
    eval_file, ids_file, *_ = make_eval(tmp_path)
    pins = {ids_file: C.sha256(ids_file), eval_file: C.sha256(eval_file)}
    old = R.N_TARGETS
    try:
        R.N_TARGETS = len(json.loads(ids_file.read_text())["ids"])
        assert R.data_problems(pins, ids_file, eval_file) == []
        assert R.data_problems({**pins, eval_file: "0" * 64}, ids_file, eval_file) == \
            [f"{eval_file} does not match its pinned SHA-256"]
        assert R.data_problems({**pins, tmp_path / "nope": "0" * 64}, ids_file, eval_file) == \
            [f"{tmp_path / 'nope'} does not exist"]
    finally:
        R.N_TARGETS = old


def test_plan_arms_are_the_fifteen_of_plan_section_3():
    arms = R.plan_arms()
    assert len(arms) == 15 and len(set(a[:3] for a in arms)) == 15
    for config in R.CONFIGS:
        assert sorted(a[2] for a in arms if a[0] == config) == sorted(C.SEEDS)
    assert all(a[1] in (None, C.PRIMARY_GAMMA) for a in arms)
    assert {a[:3] for a in arms if a[3] == "october"} == {
        *[("concept", None, s) for s in C.SEEDS], ("alt_mass", C.PRIMARY_GAMMA, 42),
        *[("alt_mass_balance", C.PRIMARY_GAMMA, s) for s in C.SEEDS]}


def test_command_is_the_round2_lexsub_command():
    tag, ckpt, out = MODELS[0], "/ckpt/x", Path("/tmp/out.json")
    mine, cwd = R.command(tag, ckpt, R.EVAL_FILE, out, "py")
    theirs, cwd2 = E.command("coinco_test", tag, ckpt, out, "py")
    i = theirs.index("--swords_json") + 1
    assert cwd == cwd2 and mine[:i] == theirs[:i] and mine[i + 1:] == theirs[i + 1:]
    assert mine[i] == R.EVAL_FILE and theirs[i] == E.LEXSUB["coinco_test"] == R.PREFLIGHT_FILE


def test_correction_note_rules(tmp_path):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    assert R.correction_problems(None) == []
    assert "must be sealed/correction_" in R.correction_problems(sealed / "fix.md", tmp_path, sealed)[0]
    assert "does not exist" in R.correction_problems(sealed / "correction_2026-10-30_gap.md", tmp_path, sealed)[0]
    assert R.output_root("m", sealed / "correction_2026-10-30_gap.md", tmp_path) == \
        tmp_path / "correction_correction_2026-10-30_gap" / "m"


# ---------- runner: scoring, retries, completion ----------

class FakeEvaluator:
    """Plays eval_swords.py: fails `fail` times (exit 1, a traceback after a score line), then succeeds."""

    def __init__(self, ids, fail=0, drop=()):
        self.ids, self.fail, self.drop, self.calls = sorted(ids), fail, drop, 0

    def __call__(self, argv, cwd, gpu, log):
        self.calls += 1
        Path(log).write_text("  left  GAP=0.4321 spearman=0.1\n")
        if self.calls <= self.fail:
            Path(log).write_text("  left  GAP=0.4321\nTraceback (most recent call last):\n  File x\nRuntimeError: CUDA out of memory\n")
            return 1
        out, ckpt = argv[argv.index("--results_json") + 1], argv[argv.index("--checkpoints") + 1]
        Path(out).write_text(json.dumps(output_record(ckpt, self.ids, lambda t, m: 0.5, self.drop)))
        return 0


def score(tmp_path, evaluator, root=None):
    root = root or tmp_path / "score" / "m"
    ids = {f"t{i}" for i in range(5)}
    arm = ("alt_mass", C.PRIMARY_GAMMA, 123)
    status = R.score_one(MODELS[0], arm, "/ckpt/a", root, ids, 1, "py", runner=evaluator,
                         data_file=tmp_path / "eval.json.gz", record=lambda *a, **k: None)
    return status, root, A.flat(*arm), ids


def test_retries_then_scores(tmp_path, capsys):
    ev = FakeEvaluator({f"t{i}" for i in range(5)}, fail=2)
    status, root, name, ids = score(tmp_path, ev)
    assert status == "ok" and ev.calls == 3
    ends = [r for r in R.read_attempts(root) if r["event"] == "end"]
    assert [r["ok"] for r in ends] == [False, False, True] and [r["attempt"] for r in ends] == [1, 2, 3]
    assert R.output_ok(root / "outputs" / f"{name}.json", "/ckpt/a", ids)
    assert not (root / "attempts" / f"{name}.attempt3.json").exists()     # became the output
    printed = capsys.readouterr().out
    assert "CUDA out of memory" in printed and "GAP=" not in printed       # traceback only, never a score


def test_three_failures_make_a_missing_run_and_no_fourth_attempt(tmp_path):
    ev = FakeEvaluator(set(), fail=99)
    status, root, name, ids = score(tmp_path, ev)
    assert status == "missing" and ev.calls == R.MAX_ATTEMPTS
    status, *_ = score(tmp_path, ev, root)
    assert status == "missing" and ev.calls == R.MAX_ATTEMPTS
    record = R.completion_record("m", root, [(("alt_mass", C.PRIMARY_GAMMA, 123), "round2", "/ckpt/a", "missing")])
    assert record["outputs"] == {} and record["missing"][name]["exit_codes"] == [1, 1, 1]


def test_interrupted_attempt_is_not_counted(tmp_path):
    root = tmp_path / "score" / "m"
    name = A.flat("alt_mass", C.PRIMARY_GAMMA, 123)
    R.append_attempt(root, {"flat": name, "event": "start", "attempt": 1, "checkpoint": "/ckpt/a", "utc": "x"})
    ev = FakeEvaluator({f"t{i}" for i in range(5)}, fail=2)
    status, *_ = score(tmp_path, ev, root)
    assert status == "ok" and ev.calls == 3
    assert [r["attempt"] for r in R.read_attempts(root) if r["event"] == "end"] == [2, 3, 4]


def test_wrong_target_set_stops_instead_of_retrying(tmp_path):
    ev = FakeEvaluator({f"t{i}" for i in range(5)}, drop=("t3",))
    with pytest.raises(R.StopRun, match="not a technical failure"):
        score(tmp_path, ev)
    root = tmp_path / "score" / "m"
    name = A.flat("alt_mass", C.PRIMARY_GAMMA, 123)
    assert ev.calls == 1 and (root / "attempts" / f"{name}.attempt1.json").is_file()
    assert not (root / "outputs" / f"{name}.json").exists()


def test_scored_checkpoint_is_not_rescored(tmp_path):
    ev = FakeEvaluator({f"t{i}" for i in range(5)})
    status, root, name, ids = score(tmp_path, ev)
    status, *_ = score(tmp_path, ev, root)
    assert status == "ok" and ev.calls == 1
    record = R.completion_record("m", root, [(("alt_mass", C.PRIMARY_GAMMA, 123), "round2", "/ckpt/a", "ok")])
    assert record["outputs"][name]["sha256"] == C.sha256(root / "outputs" / f"{name}.json")


def test_traceback_tail(tmp_path):
    log = tmp_path / "e.log"
    log.write_text("start\n  left GAP=0.4\nTraceback (most recent call last):\n  File y\nValueError: boom\n")
    assert R.traceback_tail(log) == ["Traceback (most recent call last):", "  File y", "ValueError: boom"]
    log.write_text("  left GAP=0.4\n")
    assert R.traceback_tail(log) == []


def test_preflight_comparison_is_exact():
    rec = output_record("/c", ["a", "b"], lambda t, m: 0.25)
    rec[0]["left"]["per_target"][0]["spearman"] = float("nan")
    same = json.loads(json.dumps(rec))
    assert R.compare_records(same[0], rec[0])["passed"]
    off = json.loads(json.dumps(rec))
    off[0]["full"]["per_target"][1]["gap"] += 1e-15
    result = R.compare_records(off[0], rec[0])
    assert not result["passed"] and result["problems"] == ["full b: gap"]


def test_preflight_end_to_end(tmp_path, monkeypatch):
    stored = output_record("/oct/zhang_seed42", ["a", "b", "c"], lambda t, m: 0.3)[0]

    class Src:
        def october_record(self, tag, name, label):
            assert (name, label) == ("coinco_test", "zhang_seed42")
            return stored, Path("/oct/coinco_test.json")

    monkeypatch.setattr(R, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(R, "ledger", lambda *a, **k: None)

    def rescore(argv, cwd, gpu, log, value=0.3):
        assert argv[argv.index("--swords_json") + 1] == R.PREFLIGHT_FILE
        out = argv[argv.index("--results_json") + 1]
        Path(out).write_text(json.dumps(output_record(stored["checkpoint"], ["a", "b", "c"], lambda t, m: value)))
        return 0

    tag = MODELS[0]
    assert R.preflight(tag, 1, "py", runner=rescore, src=Src())["passed"]
    assert R.preflight_status(tmp_path / tag)["passed"]
    assert not R.preflight(tag, 1, "py", runner=lambda *a: rescore(*a, value=0.31), src=Src())["passed"]
    assert not R.preflight_status(tmp_path / tag)["passed"]                  # the latest run counts
    assert len(list((tmp_path / tag / "preflight").glob("run*.result.json"))) == 2   # none overwritten
