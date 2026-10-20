"""Task 6 (Tier 1): the 2x2 effects of Alt-Mass and Alt-Balance, with bootstrap intervals and Holm.

Cells per model and gamma (Y_mb: mass on/off, balance on/off; protocol §1, §3):
  Y00 Concept loss       October zhang_seed{s}; the same three runs at both gammas
  Y10 Alt-Mass           October seed 42 (zhang_plus_mass), round-2 seeds 123 and 2024
  Y01 Alt-Balance        round 2, three seeds
  Y11 Alt-Mass+Balance   October zhang_plus_uniform (Task 1 test B licenses the reuse)
Every comparison is seed-matched and uses the items that all twelve runs of it score.

Inference: paired hierarchical bootstrap, 10,000 replicates: resample the seeds, then the
targets, with one target draw shared by every cell and seed (targets are crossed with
seeds, not nested in them); percentile 95% CI; two-sided p with the +1 correction.
Secondary: the October target-only bootstrap (per-target difference averaged over seeds,
targets resampled).  Holm within each protocol §5 family, per metric, scorer and gamma,
across contrast x model x dataset.  The opening block (SWORDS test GAP, left scorer,
gamma .125) is Holm-corrected across the three models separately for each main effect and
is marked secondary (§3).  No arm is selected and no winner declared (§7).

    python scripts/round2/analyze_round2.py [--allow-partial] [--reps 10000]
Writes results/round2/analysis/{metrics_long,cells,effects}.csv and report.md.  Refuses
while any cell lacks a seed; --allow-partial is for monitoring and marks everything PARTIAL.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

import r2common as C

BOOT_SEED = 20261017
LEXSUB = ("swords_test", "coinco_test", "semeval07_test")
SCORERS = ("left", "full")
LEXSUB_METRICS = ("gap", "auroc", "p_at_1", "alternatives_nll", "rejected_mass_share")
HARNESS_TASKS = ("arc_challenge", "arc_easy", "openbookqa", "hellaswag", "piqa", "winogrande")
HARNESS_GROUPS = {"knowledge_mean": HARNESS_TASKS[:3], "commonsense_mean": HARNESS_TASKS[3:]}
OCT_DIRS = ("task15b_screen_test", "task15b_screen_shared_gamma", "task15b_screen_confirm")
CELL_ORDER = ("concept", "alt_mass", "alt_balance", "alt_mass_balance")     # Y00 Y10 Y01 Y11
WHOLESET = "concept_uncapped"                                                # Concept loss + whole-set (T2-c)
WS_ORDER = ("concept", "alt_mass", WHOLESET)                                 # its own design, primary gamma only
ALL_ARMS = CELL_ORDER + (WHOLESET,)

# metric -> (label, display scale, better)
METRICS = {
    "gap": ("GAP", 100, "higher"), "auroc": ("AUROC", 100, "higher"), "p_at_1": ("p@1", 100, "higher"),
    "alternatives_nll": ("accepted-alternative NLL", 1, "lower"),
    "rejected_mass_share": ("rejected share", 100, "lower"),
    "observed_share": ("observed share", 100, ""),
    "global_nll": ("C4 test NLL", 1, "lower"), "content_nll": ("C4 test content-word NLL", 1, "lower"),
    "accuracy": ("bm-semlex accuracy", 100, "higher"), "margin": ("bm-semlex mean margin", 1, "higher"),
    "sts_mean": ("STS mean Spearman (9 tasks)", 1, "higher"),
    "acc": ("harness accuracy", 100, "higher"),
    "knowledge_mean": ("harness knowledge mean", 100, "higher"),
    "commonsense_mean": ("harness commonsense mean", 100, "higher"),
    # C4 training-distribution diagnostic (protocol §0): descriptive, so no direction is better.
    "set_mass": ("C4 M(C)", 100, ""), "alternative_mass": ("C4 M(A)", 100, ""),
    "alternative_balance_kl": ("C4 KL(u_A||q_A)", 1, ""),
}
# (family, key, display, weights over CELL_ORDER)
CONTRASTS = (
    ("F5", "mass_main", "Mass main effect", (-.5, .5, -.5, .5)),
    ("F5", "balance_main", "Balance main effect", (-.5, -.5, .5, .5)),
    ("F5", "interaction", "Mass x Balance interaction", (1, -1, -1, 1)),
    ("F2", "mb_vs_mass", "Alt-Mass+Balance - Alt-Mass", (0, -1, 0, 1)),
    ("F2", "mb_vs_balance", "Alt-Mass+Balance - Alt-Balance", (0, 0, -1, 1)),
    ("F2", "mass_vs_balance", "Alt-Mass - Alt-Balance", (0, 1, -1, 0)),
    ("F1", "mass_vs_concept", "Alt-Mass - Concept loss", (-1, 1, 0, 0)),
    ("F1", "balance_vs_concept", "Alt-Balance - Concept loss", (-1, 0, 1, 0)),
    ("F1", "mb_vs_concept", "Alt-Mass+Balance - Concept loss", (-1, 0, 0, 1)),
)
# Whole-set control (protocol v2 §4, §5 families 1 and 3), weights over WS_ORDER.  Computed on
# its own three-arm design so that the 2x2 effects above are unchanged by its presence.
WS_CONTRASTS = (
    ("F3", "mass_vs_wholeset", "Alt-Mass - Concept loss + whole-set", (0, 1, -1)),
    ("F1", "wholeset_vs_concept", "Concept loss + whole-set - Concept loss", (-1, 0, 1)),
)
FAMILY_NAMES = {"primary": "opening block (secondary)", "F1": "each arm vs Concept loss",
                "F2": "among the alternative arms", "F3": "Alt-Mass vs Concept loss + whole-set",
                "F5": "factorial effects"}
HEADLINE = [(b, s, "gap") for b in LEXSUB for s in SCORERS] + [
    ("swords_test", "left", "alternatives_nll"), ("swords_test", "left", "rejected_mass_share"),
    ("sts", "", "sts_mean"), ("c4_test", "", "global_nll"), ("bm_semlex", "left", "accuracy"),
    ("harness", "", "knowledge_mean"), ("harness", "", "commonsense_mean")]
DIAGNOSTIC = [("c4_concept_sets", "", m) for m in
              ("alternative_mass", "set_mass", "observed_share", "alternative_balance_kl")]


class Missing(Exception):
    pass


def flat(objective, gamma, seed):
    """Concept loss has no gamma; its files carry g0, as C.run_id does."""
    return f"{objective}_g{gamma or 0:g}_seed_{seed}"


def needed_arms():
    """(objective, gamma, seed, source); gamma None for Concept loss."""
    arms = [("concept", None, s, "october") for s in C.SEEDS]
    for g in C.MAIN_GAMMAS:
        arms += [("alt_mass", g, s, "october" if s == C.SEEDS[0] else "round2") for s in C.SEEDS]
        arms += [("alt_balance", g, s, "round2") for s in C.SEEDS]
        arms += [("alt_mass_balance", g, s, "october") for s in C.SEEDS]
    return arms


def wholeset_arms():
    return [(WHOLESET, C.PRIMARY_GAMMA, s, "round2") for s in C.SEEDS]


def wholeset_started(tag, src):
    """True once any whole-set run of this model has a test-pass output; then every one is required."""
    return any((src.r2 / "eval" / tag / step / f"{flat(*a[:3])}.json").is_file()
               for a in wholeset_arms() for step in LEXSUB)


def october_label(objective, gamma, seed):
    if objective == "concept":
        return f"zhang_seed{seed}"
    kind = {"alt_mass": "mass", "alt_mass_balance": "uniform"}[objective]
    return f"zhang_plus_{kind}_g{gamma:g}_seed{seed}"


def read_json(path):
    if not Path(path).is_file():
        raise Missing(str(path))
    return json.loads(Path(path).read_text())


def check_aggregate(values, stored, what):
    vals = [v for v in values if v is not None]
    if stored is None or not vals:
        return
    if abs(sum(vals) / len(vals) - stored) > 1e-9:
        sys.exit(f"per-item values do not reproduce the stored aggregate: {what}")


class Sources:
    """Locate one run's output for one benchmark, October or round 2."""

    def __init__(self, outputs, r2):
        self.outputs, self.r2 = Path(outputs), Path(r2)
        self._cache = {}

    def locked(self, tag):
        return read_json(self.outputs / tag / "task15b_screen_test" / "swords_test_locked.json")["arms"]

    def by_checkpoint(self, path):
        if path not in self._cache:
            self._cache[path] = {r["checkpoint"]: r for r in read_json(path)}
        return self._cache[path]

    def october_record(self, tag, name, label):
        ckpt = self.locked(tag).get(label)
        if not ckpt:
            raise Missing(f"{tag}: {label} is not among the locked October arms")
        dirs = OCT_DIRS[:1] if name in LEXSUB else OCT_DIRS
        for d in dirs:
            f = self.outputs / tag / d / f"{name}.json"
            if f.is_file() and ckpt in self.by_checkpoint(f):
                return self.by_checkpoint(f)[ckpt], f
        raise Missing(f"{tag}: no October {name}.json scores {label}")

    def round2_record(self, tag, step, arm):
        f = self.r2 / "eval" / tag / step / f"{flat(*arm)}.json"
        rows = read_json(f)
        want = f"{tag}/{arm[0]}/g{arm[1]:g}/seed_{arm[2]}"
        if len(rows) != 1 or not rows[0]["checkpoint"].rstrip("/").endswith(want):
            sys.exit(f"{f} does not hold exactly the checkpoint {want}")
        return rows[0], f

    def concept_sets(self, tag, arm, source):
        """C4 concept sets: every arm, October ones included, re-scored by c4_diagnostic.py."""
        root = self.r2 / "eval" / tag / "concept_sets"
        f = root / f"{flat(*arm)}.json"
        rows = read_json(f)
        if source == "october":
            want = self.locked(tag).get(october_label(*arm))
            ok = len(rows) == 1 and rows[0]["checkpoint"] == want
        else:
            want = f"{tag}/{arm[0]}/g{arm[1]:g}/seed_{arm[2]}"
            ok = len(rows) == 1 and rows[0]["checkpoint"].rstrip("/").endswith(want)
        if not ok:
            sys.exit(f"{f} does not hold exactly the checkpoint {want}")
        if source == "october" and not read_json(root / "reproduction.json").get("passed"):
            sys.exit(f"{root / 'reproduction.json'}: the October fields did not reproduce")
        return rows[0], f

    def record(self, tag, name, arm, source):
        if source == "october":
            return self.october_record(tag, name, october_label(*arm))
        return self.round2_record(tag, "c4_test" if name == "perplexity" else name, arm)

    def sts(self, tag, arm, source):
        if source == "october":
            name = "sts_" + october_label(*arm).replace("_", " ") + ".csv"
            hits = [self.outputs / tag / d / name for d in OCT_DIRS if (self.outputs / tag / d / name).is_file()]
            if not hits:
                raise Missing(f"{tag}: no October {name}")
            f = hits[0]
        else:
            f = self.r2 / "eval" / tag / "sts" / f"{flat(*arm)}.csv"
            if not f.is_file():
                raise Missing(str(f))
        with open(f, newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("eval_split", "test") == "test"]
        if len(rows) != 9:
            sys.exit(f"{f}: expected 9 STS tasks, found {len(rows)}")
        return {("", r["eval_task"], ""): float(r["main_score"]) for r in rows}, f

    def harness(self, tag, arm, source):
        if source == "october":
            label, f = october_label(*arm), self.outputs / tag / "harness_confirm.json"
        else:
            label, f = flat(*arm), self.r2 / "eval" / tag / "harness.json"
        results = read_json(f)
        if label not in results:
            raise Missing(f"{f} has no {label}")
        rec = results[label]
        if rec.get("_limit") is not None or not rec.get("_log_samples"):
            sys.exit(f"{f} {label}: not a full pass with per-question samples")
        raw = f.parent / f"{f.stem}_raw" / label
        items = {}
        for task in HARNESS_TASKS:
            metric = "acc_norm" if any(k.startswith("acc_norm,") for k in rec[task]) else "acc"
            files = sorted(glob.glob(str(raw / "**" / f"samples_{task}_*.jsonl"), recursive=True),
                           key=lambda p: Path(p).name)
            if not files:
                raise Missing(f"{raw}: no samples for {task}")
            vals = {}
            for line in Path(files[-1]).read_text().splitlines():
                if line.strip():
                    s = json.loads(line)
                    vals[(task, str(s["doc_id"]), "")] = float(s[metric])
            check_aggregate(list(vals.values()), rec[task][f"{metric},none"], f"{label} {task}")
            items.update(vals)
        return items, f


def load_model(tag, src):
    """(bench, scorer, metric) -> {arm: {item: value}}, files, missing, notes."""
    data, files, missing, notes = defaultdict(dict), {}, [], []
    arms = needed_arms()
    if wholeset_started(tag, src):
        arms += wholeset_arms()
    else:
        notes.append(f"{tag}: no whole-set (T2-c) run scored yet; Family 3 omitted")

    def put(bench, scorer, metric, arm, values):
        data[(bench, scorer, metric)][arm[:3]] = values

    for objective, gamma, seed, source in arms:
        arm = (objective, gamma, seed)
        for bench in LEXSUB:
            try:
                rec, f = src.record(tag, bench, arm, source)
            except Missing as e:
                missing.append(str(e))
                continue
            files[(bench, arm)] = f, source
            for scorer in SCORERS:
                per = rec[scorer]["per_target"]
                for m in LEXSUB_METRICS:
                    vals = {("", t["target_id"], ""): t[m] for t in per}
                    check_aggregate(list(vals.values()), rec[scorer][m], f"{f} {scorer} {m}")
                    put(bench, scorer, m, arm, vals)
                put(bench, scorer, "observed_share", arm,
                    {("", t["target_id"], ""): (math.exp(t["inclusive_nll"] - t["gold_nll"])
                                                if t["inclusive_nll"] is not None and t["gold_nll"] is not None
                                                else None) for t in per})
        try:
            rec, f = src.record(tag, "perplexity", arm, source)
            files[("c4_test", arm)] = f, source
            put("c4_test", "", "global_nll", arm, {("", "global", ""): rec["global"]["mean_nll"]})
            put("c4_test", "", "content_nll", arm, {("", "content_words", ""): rec["content_words"]["mean_nll"]})
        except Missing as e:
            missing.append(str(e))
        try:
            rec, f = src.record(tag, "bm_semlex", arm, source)
            files[("bm_semlex", arm)] = f, source
            for scorer in SCORERS:
                rows = rec[scorer]["per_row"]
                acc = {("", r["row_id"], ""): float(r["correct"]) for r in rows}
                mar = {("", r["row_id"], ""): r["margin"] for r in rows}
                check_aggregate(list(acc.values()), rec[scorer]["accuracy"], f"{f} {scorer} accuracy")
                check_aggregate(list(mar.values()), rec[scorer]["mean_margin"], f"{f} {scorer} margin")
                put("bm_semlex", scorer, "accuracy", arm, acc)
                put("bm_semlex", scorer, "margin", arm, mar)
        except Missing as e:
            missing.append(str(e))
        try:
            vals, f = src.sts(tag, arm, source)
            files[("sts", arm)] = f, source
            put("sts", "", "sts_mean", arm, vals)
        except Missing as e:
            missing.append(str(e))
        try:
            items, f = src.harness(tag, arm, source)
            files[("harness", arm)] = f, source
            for task in HARNESS_TASKS:
                put(f"harness_{task}", "", "acc", arm, {k: v for k, v in items.items() if k[0] == task})
            for group, tasks in HARNESS_GROUPS.items():
                put("harness", "", group, arm, {k: v for k, v in items.items() if k[0] in tasks})
        except Missing as e:
            missing.append(str(e))

    # C4 training-distribution diagnostic (protocol §0): written by c4_diagnostic.py for every
    # arm, October ones re-scored; once any file of this model exists, every arm is required.
    # Items are slots, clustered by C4 sequence (row) in the bootstrap.
    if not any((src.r2 / "eval" / tag / "concept_sets" / f"{flat(*a[:3])}.json").is_file() for a in arms):
        notes.append(f"{tag}: C4 diagnostic (c4_diagnostic.py) not run yet; omitted")
    else:
        for objective, gamma, seed, source in arms:
            arm = (objective, gamma, seed)
            try:
                rec, f = src.concept_sets(tag, arm, source)
            except Missing as e:
                missing.append(str(e))
                continue
            files[("c4_concept_sets", arm)] = f, source
            for m in ("set_mass", "alternative_balance_kl"):
                check_aggregate([s[m] for s in rec["per_slot"]], rec[m], f"{f} {m}")
            # All four over the slots with at least one alternative, where each is defined
            # (with |A| = 0 the observed share is 1 in every arm).
            slots = [s for s in rec["per_slot"] if s["alternative_nll"] is not None]
            key = lambda s: ("", str(s["row_id"]), str(s["slot_id"]))
            put("c4_concept_sets", "", "alternative_mass", arm,
                {key(s): math.exp(-s["alternative_nll"]) for s in slots})
            put("c4_concept_sets", "", "set_mass", arm, {key(s): s["set_mass"] for s in slots})
            put("c4_concept_sets", "", "observed_share", arm,
                {key(s): math.exp(s["set_nll"] - s["observed_target_nll"]) for s in slots})
            put("c4_concept_sets", "", "alternative_balance_kl", arm,
                {key(s): s["alternative_balance_kl"] for s in slots})
    return data, files, missing, notes


def finite(v):
    return v is not None and math.isfinite(v)


def stat(values):
    """Mean over strata of the item mean within each stratum (the evaluator's aggregate)."""
    by = defaultdict(list)
    for (stratum, _, _), v in values.items():
        if finite(v):
            by[stratum].append(v)
    if not by:
        return None, 0
    return float(np.mean([np.mean(v) for v in by.values()])), sum(len(v) for v in by.values())


def design(per_arm, gamma, seeds, order=CELL_ORDER):
    """Unit-level sums X[cell, seed, unit], item counts n[unit], strata index arrays."""
    keys = [(c, None if c == "concept" else gamma, s) for c in order for s in seeds]
    common = set.intersection(*({k for k, v in per_arm[a].items() if finite(v)} for a in keys))
    items = sorted(common)
    units = sorted({it[:2] for it in items})
    uidx = {u: i for i, u in enumerate(units)}
    X = np.zeros((len(order), len(seeds), len(units)))
    n = np.zeros(len(units))
    cols = np.array([uidx[it[:2]] for it in items], dtype=int)
    np.add.at(n, cols, 1)
    for ci, c in enumerate(order):
        for si, s in enumerate(seeds):
            vals = per_arm[(c, None if c == "concept" else gamma, s)]
            np.add.at(X[ci, si], cols, [vals[it] for it in items])
    strata = [np.array([i for i, u in enumerate(units) if u[0] == st])
              for st in sorted({u[0] for u in units})]
    return X, n, strata, len(items)


def point(D, n, strata):
    """D: [K, S, U] unit sums -> [K] estimate (mean over seeds of the stratified mean)."""
    return np.mean([D[:, :, idx].sum(-1) / n[idx].sum() for idx in strata], axis=0).mean(-1)


def bootstrap(D, n, strata, reps, key, chunk=500):
    """[reps, K] hierarchical (seeds, then shared targets) and target-only replicates."""
    K, S, U = D.shape
    rng = np.random.default_rng([BOOT_SEED, zlib.crc32(key.encode())])
    flatD, Dbar = D.reshape(K * S, U), D.mean(1)
    hier, tonly = np.empty((reps, K)), np.empty((reps, K))
    for start in range(0, reps, chunk):
        b = min(chunk, reps - start)
        draw = rng.integers(0, S, (b, S))
        w = np.bincount((draw + np.arange(b)[:, None] * S).ravel(), minlength=b * S).reshape(b, S) / S
        per_seed, per_bar = np.zeros((b, K, S)), np.zeros((b, K))
        for idx in strata:
            m = len(idx)
            pick = rng.integers(0, m, (b, m))
            cnt = np.bincount((pick + np.arange(b)[:, None] * m).ravel(), minlength=b * m).reshape(b, m).astype(float)
            den = cnt @ n[idx]
            per_seed += (cnt @ flatD[:, idx].T).reshape(b, K, S) / den[:, None, None]
            per_bar += (cnt @ Dbar[:, idx].T) / den[:, None]
        hier[start:start + b] = np.einsum("bks,bs->bk", per_seed, w) / len(strata)
        tonly[start:start + b] = per_bar / len(strata)
    return hier, tonly


ZERO_TOL = 1e-12   # far below any real replicate (0/1 metrics move in steps of ~1e-4)


def p_two_sided(reps):
    # A 0/1 metric gives many replicates that are exactly zero; summation order turns them into
    # +-1e-17 differently per platform, which moved p by up to 0.16 between the server and a laptop.
    reps = np.where(np.abs(reps) < ZERO_TOL, 0.0, reps)
    B = len(reps)
    return min(1.0, 2 * min((1 + np.sum(reps <= 0)) / (B + 1), (1 + np.sum(reps >= 0)) / (B + 1)))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def analyse(models, reps, partial):
    cells, effects, longs = [], [], []
    for tag, (data, files) in models.items():
        for (bench, scorer, metric), per_arm in sorted(data.items(), key=lambda kv: kv[0]):
            group = "harness" if bench.startswith("harness") else bench
            for arm, values in sorted(per_arm.items(), key=lambda kv: (ALL_ARMS.index(kv[0][0]), kv[0][1] or 0, kv[0][2])):
                v, k = stat(values)
                f, source = files[(group, arm)]
                longs.append({"model": tag, "benchmark": bench, "scorer": scorer, "metric": metric,
                              "objective": arm[0], "arm": C.DISPLAY[arm[0]],
                              "gamma": "" if arm[1] is None else f"{arm[1]:g}", "seed": arm[2],
                              "source": source, "value": v, "n_items": k, "file": str(f)})
            # (gamma, arm order, contrasts, which cells to write): the 2x2 at both gammas, then the
            # whole-set design at the primary gamma, whose concept and Alt-Mass cells are already written.
            blocks = [(g, CELL_ORDER, CONTRASTS, CELL_ORDER) for g in C.MAIN_GAMMAS]
            blocks.append((C.PRIMARY_GAMMA, WS_ORDER, WS_CONTRASTS, (WHOLESET,)))
            for gamma, order, contrasts, write_cells in blocks:
                seeds = [s for s in C.SEEDS if all((c, None if c == "concept" else gamma, s) in per_arm
                                                   for c in order)]
                if not seeds:
                    continue
                X, n, strata, n_items = design(per_arm, gamma, seeds, order)
                if n_items == 0:
                    continue
                for ci, c in enumerate(order):
                    if c not in write_cells:
                        continue
                    per_seed = [float(np.mean([X[ci, si, idx].sum() / n[idx].sum() for idx in strata]))
                                for si in range(len(seeds))]
                    cells.append({"model": tag, "benchmark": bench, "scorer": scorer, "metric": metric,
                                  "gamma": f"{gamma:g}", "objective": c, "arm": C.DISPLAY[c],
                                  "mean": float(np.mean(per_seed)),
                                  "sd_seeds": float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else "",
                                  "per_seed": " ".join(f"{s}:{v:.6g}" for s, v in zip(seeds, per_seed)),
                                  "n_items": n_items, "n_seeds": len(seeds)})
                W = np.array([w for *_, w in contrasts], dtype=float)
                D = np.einsum("kc,csu->ksu", W, X)
                est = point(D, n, strata)
                key = f"{tag}|{bench}|{scorer}|{metric}|{gamma:g}" + ("|wholeset" if order is WS_ORDER else "")
                hier, tonly = bootstrap(D, n, strata, reps, key)
                single = all(len(idx) == 1 for idx in strata)
                for k, (family, ckey, cname, _) in enumerate(contrasts):
                    lo, hi = np.percentile(hier[:, k], [2.5, 97.5])
                    tlo, thi = (np.percentile(tonly[:, k], [2.5, 97.5]) if not single else (None, None))
                    effects.append({"family": family, "model": tag, "benchmark": bench, "scorer": scorer,
                                    "metric": metric, "gamma": f"{gamma:g}", "contrast": ckey,
                                    "contrast_name": cname, "estimate": float(est[k]),
                                    "ci_low": float(lo), "ci_high": float(hi), "p": p_two_sided(hier[:, k]),
                                    "to_ci_low": tlo, "to_ci_high": thi,
                                    "to_p": None if single else p_two_sided(tonly[:, k]),
                                    "n_items": n_items, "n_units": int(len(n)), "n_seeds": len(seeds),
                                    "bootstrap": "seeds only (one aggregate per run)" if single
                                    else "seeds, then targets", "better": METRICS[metric][2],
                                    "partial": partial})
    primary = [dict(e, family="primary") for e in effects
               if e["family"] == "F5" and e["contrast"] in ("mass_main", "balance_main")
               and (e["benchmark"], e["scorer"], e["metric"], e["gamma"]) == ("swords_test", "left", "gap", f"{C.PRIMARY_GAMMA:g}")]
    effects += primary
    groups = defaultdict(list)
    for e in effects:
        key = ((e["family"], e["contrast"]) if e["family"] == "primary"
               else (e["family"], e["metric"], e["scorer"], e["gamma"]))
        groups[key].append(e)
    for members in groups.values():
        for col, pcol in (("p_holm", "p"), ("to_p_holm", "to_p")):
            ok = [e for e in members if e[pcol] is not None]
            for e, adj in zip(ok, holm([e[pcol] for e in ok])):
                e[col] = adj
                e[col.replace("p_holm", "holm_m")] = len(ok)
    return longs, cells, effects


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(metric, v, signed=False):
    if v is None or v == "":
        return ""
    scale = METRICS[metric][1]
    digits = 2 if scale == 100 else 3
    return f"{v * scale:+.{digits}f}" if signed else f"{v * scale:.{digits}f}"


def cell_text(e):
    mark = " †" if e.get("p_holm") is not None and e["p_holm"] < 0.05 else ""
    return (f"{fmt(e['metric'], e['estimate'], True)} [{fmt(e['metric'], e['ci_low'], True)}, "
            f"{fmt(e['metric'], e['ci_high'], True)}] p={e['p']:.4f}, Holm {e.get('p_holm', float('nan')):.4f}{mark}")


def label(bench, scorer, metric):
    name, scale, better = METRICS[metric]
    unit = " (x100)" if scale == 100 else ""
    return f"{bench}{' ' + scorer if scorer else ''} {name}{unit}" + (f", {better} is better" if better else "")


def report(path, cells, effects, missing, notes, harness_check, partial, reps):
    L = []
    L.append("# Round 2 (Tier 1): Alt-Mass and Alt-Balance, factorial effects\n")
    if partial:
        L.append("> **PARTIAL — monitoring only.** Some cells lack seeds; nothing here is a result.\n")
    L.append(f"Generated {C.now_utc()} at git {C.git_commit()}, under `ROUND2_PROTOCOL.md` "
             f"{protocol_version()}. Every new arm was scored on test sets that were read before it existed "
             "(ledger `added_after_first_test_read=true`), so everything below is secondary by "
             "construction. No arm is selected and no winner is declared (protocol §7).\n")
    L.append(f"Inference: paired hierarchical bootstrap, {reps:,} replicates (seeds resampled, then "
             "targets, one target draw shared by every cell and seed); percentile 95% CI; two-sided p "
             "with the +1 correction. Holm within each §5 family per metric, scorer and γ, across "
             "contrast × model × dataset. † marks Holm-adjusted p < 0.05. Effects are differences "
             "(arm − reference) in the metric's own direction; see each table's header.\n")
    L.append("## Opening block — SECONDARY (protocol §3)\n")
    L.append(f"Mass and Balance main effects on **SWORDS test GAP, left scorer, γ = {C.PRIMARY_GAMMA:g}**, "
             "per model; Holm across the three models separately for each main effect. The protocol's "
             "primary endpoint is the sealed holdout, which this round does not score (§0, §3).\n")
    hdr = "| Model | n targets | " + " | ".join(C.DISPLAY[c] for c in CELL_ORDER) + " | Mass main effect | Balance main effect |"
    L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    g = f"{C.PRIMARY_GAMMA:g}"
    for tag in C.FAMILIES:
        cm = {c["objective"]: c for c in cells if (c["model"], c["benchmark"], c["scorer"], c["metric"], c["gamma"])
              == (tag, "swords_test", "left", "gap", g)}
        ef = {e["contrast"]: e for e in effects if e["family"] == "primary" and e["model"] == tag}
        if not cm or not ef:
            L.append(f"| {tag} | — | not available |")
            continue
        L.append(f"| {tag} | {cm['concept']['n_items']} | " +
                 " | ".join(f"{fmt('gap', cm[c]['mean'])} ± {fmt('gap', cm[c]['sd_seeds'] or 0)}" for c in CELL_ORDER) +
                 f" | {cell_text(ef['mass_main'])} | {cell_text(ef['balance_main'])} |")
    L.append("\nCell entries: mean ± SD over the three seeds, GAP × 100.\n")

    L.append("## Eligible items\n")
    L.append("Items every run of a comparison scores (targets; bm-semlex rows; STS tasks; harness "
             "questions; C4 NLL is one aggregate per run). Accepted-alternative NLL and rejected "
             "share drop targets with prefix collisions.\n")
    counts = defaultdict(dict)
    for c in cells:
        if c["objective"] == "concept":
            counts[(c["benchmark"], c["scorer"], c["metric"])].setdefault(c["model"], set()).add(c["n_items"])
    L += ["| benchmark | scorer | metric | " + " | ".join(C.FAMILIES) + " |", "|---|---|---|" + "---|" * len(C.FAMILIES)]
    for (b, s, m), by in sorted(counts.items()):
        L.append(f"| {b} | {s} | {m} | " + " | ".join("/".join(map(str, sorted(by.get(t, [])))) or "—" for t in C.FAMILIES) + " |")

    for gamma in (f"{C.PRIMARY_GAMMA:g}", f"{C.SENSITIVITY_GAMMA:g}"):
        cols = ALL_ARMS if any(c["objective"] == WHOLESET and c["gamma"] == gamma for c in cells) else CELL_ORDER
        L.append(f"\n## Cell means, γ = {gamma} (headline metrics; all metrics in cells.csv)\n")
        L += ["| benchmark | model | " + " | ".join(C.DISPLAY[c] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
        for b, s, m in HEADLINE + DIAGNOSTIC:
            for tag in C.FAMILIES:
                cm = {c["objective"]: c for c in cells if (c["model"], c["benchmark"], c["scorer"], c["metric"], c["gamma"]) == (tag, b, s, m, gamma)}
                if cm:
                    L.append(f"| {label(b, s, m)} | {tag} | " + " | ".join(
                        f"{fmt(m, cm[c]['mean'])} ± {fmt(m, cm[c]['sd_seeds'] or 0)}" if c in cm else "—"
                        for c in cols) + " |")
        for family in ("F5", "F2", "F1", "F3"):
            names = [c for c in CONTRASTS + WS_CONTRASTS if c[0] == family]
            have = {e["contrast"] for e in effects if e["family"] == family and e["gamma"] == gamma}
            names = [c for c in names if c[1] in have]
            if not names:
                continue
            L.append(f"\n### Family {family[1:]} ({FAMILY_NAMES[family]}), γ = {gamma}\n")
            L += ["| benchmark | model | " + " | ".join(n[2] for n in names) + " |", "|---|---|" + "---|" * len(names)]
            for b, s, m in HEADLINE + DIAGNOSTIC:
                for tag in C.FAMILIES:
                    ef = {e["contrast"]: e for e in effects if e["family"] == family and
                          (e["model"], e["benchmark"], e["scorer"], e["metric"], e["gamma"]) == (tag, b, s, m, gamma)}
                    if ef:
                        L.append(f"| {label(b, s, m)} | {tag} | " + " | ".join(cell_text(ef[n[1]]) for n in names) + " |")
    L.append("\n## Harness: Untouched re-scored in round 2 vs October\n")
    if harness_check:
        L += ["| model | task | October | round 2 | difference |", "|---|---|---|---|---|"]
        for tag, task, a, b in harness_check:
            L.append(f"| {tag} | {task} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    else:
        L.append("No round-2 harness file with an Untouched row yet.")
    L.append("\n## Not in this report\n")
    L.append("- Family 4 (Member-wise, Iyer et al.): not in the experiment list (§0). The sealed holdout: "
             "scored once at the end, by its own evaluator, not here (§8).")
    L.append("- Open-vocabulary Hit@50: dropped in protocol version 3 (no evaluator; not in the experiment list).")
    L.append("- C4 diagnostic rows (M(A), M(C), observed share, KL(u_A‖q_A)) describe the training "
             "distribution: held-out C4 slots, alternatives extracted automatically, not judged by people. "
             "All four are means over the slots with |A| ≥ 1; M(A) and M(C) are probabilities, and KL is 0 "
             "when |A| = 1.")
    L.append("- Observed share on lexsub is p(target) / (p(target) + M(accepted alternatives)), from the "
             "evaluator's per-target gold and inclusive NLLs.")
    L += [f"- {n}" for n in notes]
    if missing:
        L.append(f"\n## Missing inputs ({len(missing)})\n")
        L += [f"- {m}" for m in sorted(set(missing))]
    Path(path).write_text("\n".join(L) + "\n")


def protocol_version():
    text = (C.MAIN / "ROUND2_PROTOCOL.md").read_text().splitlines()
    line = next((l for l in text if l.startswith("- Protocol version:")), "")
    return "version " + line.split(":", 1)[1].strip() if line else "(version line missing)"


def harness_untouched(src, tag):
    try:
        old = read_json(src.outputs / tag / "harness_confirm.json").get("pretrained")
        new = read_json(src.r2 / "eval" / tag / "harness.json").get("pretrained")
    except Missing:
        return []
    if not old or not new:
        return []
    rows = []
    for task in HARNESS_TASKS:
        metric = "acc_norm,none" if "acc_norm,none" in old[task] else "acc,none"
        rows.append((tag, task, old[task][metric], new[task][metric]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=10_000)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--r2", default=str(C.R2), help="round-2 results root (holds eval/)")
    ap.add_argument("--outputs", default=str(C.OUTPUTS), help="October outputs root (read-only)")
    ap.add_argument("--out", default=None, help="default: <r2>/analysis")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing analysis in --out")
    args = ap.parse_args()
    src = Sources(Path(args.outputs), Path(args.r2))
    models, missing, notes, check = {}, [], [], []
    for tag in C.FAMILIES:
        data, files, miss, note = load_model(tag, src)
        models[tag] = (data, files)
        missing += miss
        notes += note
        check += harness_untouched(src, tag)
    if missing and not args.allow_partial:
        print(f"refusing: {len(missing)} input(s) missing; every cell needs all three seeds "
              "(--allow-partial for monitoring):")
        for m in sorted(set(missing)):
            print("  " + m)
        sys.exit(1)
    out = Path(args.out) if args.out else Path(args.r2) / "analysis"
    if (out / "effects.csv").exists() and not args.overwrite:
        sys.exit(f"{out} already holds an analysis; pass a new --out (or --overwrite to replace it)")
    out.mkdir(parents=True, exist_ok=True)
    longs, cells, effects = analyse(models, args.reps, bool(missing))
    write_csv(out / "metrics_long.csv", longs)
    write_csv(out / "cells.csv", cells)
    write_csv(out / "effects.csv", effects)
    report(out / "report.md", cells, effects, missing, notes, check, bool(missing), args.reps)
    hits = C.scan_assets(out)
    if hits:
        sys.exit(f"forbidden terms in the outputs: {hits}")
    print(f"{len(longs)} run-metric rows, {len(effects)} effects -> {out}" + (" (PARTIAL)" if missing else ""))
    for e in effects:
        if e["family"] == "primary" or (e["family"] == "F3" and (e["benchmark"], e["scorer"], e["metric"])
                                         == ("swords_test", "left", "gap")):
            print(f"  {e['model']:16s} {e['contrast_name']:36s} {cell_text(e)}")


if __name__ == "__main__":
    main()
