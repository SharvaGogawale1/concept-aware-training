#!/usr/bin/env python3
"""Qualitative SWORDS appendix: where the calibrated arm re-ranks and Zhang does not.

Reads the per-target records `eval_swords.py` already wrote, so this costs no GPU.
Emits a LaTeX fragment plus one figure, and prints the flip counts both directions --
the reverse flips are reported too, because an appendix that shows only the wins is
the thing a reviewer checks for first.

    python scripts/build_qualitative_appendix.py \
        --screen outputs/llama-3.2-1b/task15b_screen_2026-09-21 \
        --swords data/swords/swords-v1.1_dev.json.gz --outdir latex
"""
import argparse, gzip, json, textwrap
from pathlib import Path

CONSERVATIVE, LENIENT = 0.5, 0.1     # SWORDS thresholds, as in eval_swords.py
ARMS = {"ntp seed42": "NTP",
        "zhang seed42": "Set marginal ($\\gamma=0$)",
        "alternative uniform seed42": "Alternative-only ($\\gamma=1$)",
        "zhang plus uniform g0.25 seed42": "Calibrated ($\\gamma=0.25$)"}

p = argparse.ArgumentParser()
p.add_argument("--screen", required=True)
p.add_argument("--swords", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--mode", default="left", choices=["left", "full"])
p.add_argument("--n-examples", type=int, default=8)
a = p.parse_args()
screen, outdir = Path(a.screen), Path(a.outdir)

# ---- per-target metrics, keyed by arm label -----------------------------------
manifest = json.loads((screen / "manifest.json").read_text())
label_of = {str(v): k for k, v in manifest.items()}
per_target, headline = {}, {}
for entry in json.loads((screen / "swords.json").read_text()):
    label = label_of.get(str(entry["checkpoint"]))
    if label not in ARMS:
        continue
    block = entry[a.mode]
    per_target[label] = {r["target_id"]: r for r in block["per_target"]}
    headline[label] = {k: block[k] for k in ("gap", "auroc", "p_at_1")}
missing = [k for k in ARMS if k not in per_target]
assert not missing, f"arms absent from swords.json: {missing}"

# ---- SWORDS source: context, target, substitutes, human labels -----------------
with gzip.open(a.swords, "rt", encoding="utf-8") as handle:
    data = json.load(handle)
subs = {}
for sid, sub in data["substitutes"].items():
    labels = data["substitute_labels"][sid]
    if not labels:
        continue
    score = sum(x == "TRUE" for x in labels) / len(labels)
    subs.setdefault(sub["target_id"], []).append((sub["substitute"], score))

ZHANG, HYBRID = "zhang seed42", "zhang plus uniform g0.25 seed42"
common = set(per_target[ZHANG]) & set(per_target[HYBRID])

# ---- flip counts, both directions ---------------------------------------------
won = [t for t in common if per_target[ZHANG][t]["p_at_1"] == 0 and per_target[HYBRID][t]["p_at_1"] == 1]
lost = [t for t in common if per_target[ZHANG][t]["p_at_1"] == 1 and per_target[HYBRID][t]["p_at_1"] == 0]
better = sum(per_target[HYBRID][t]["gap"] > per_target[ZHANG][t]["gap"] for t in common)
worse = sum(per_target[HYBRID][t]["gap"] < per_target[ZHANG][t]["gap"] for t in common)
print(f"targets {len(common)} | top-1 flips: won {len(won)}, lost {len(lost)}"
      f" | per-target GAP better {better}, worse {worse}, tied {len(common)-better-worse}")

# ---- pick examples: top-1 flips, biggest GAP gain, short contexts, POS-diverse --
def context_of(tid):
    target = data["targets"][tid]
    ctx = data["contexts"][target["context_id"]]["context"]
    off, word = int(target["offset"]), target["target"]
    return " ".join(ctx.split()), word, off, ctx

# Every top-1 flip, in BOTH directions. There are few enough to list exhaustively, and a
# complete list cannot be accused of selection -- which a "best examples" table can.
chosen = ([(t, "gain") for t in sorted(won, key=lambda t: per_target[ZHANG][t]["gap"] - per_target[HYBRID][t]["gap"])]
          + [(t, "loss") for t in sorted(lost, key=lambda t: per_target[HYBRID][t]["gap"] - per_target[ZHANG][t]["gap"])])

# ---- figure: per-target GAP, set marginal vs calibrated ------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
x = [per_target[ZHANG][t]["gap"] for t in common]
y = [per_target[HYBRID][t]["gap"] for t in common]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
ax1.scatter(x, y, s=9, alpha=.45, color="#2b5d8a", edgecolors="none")
ax1.plot([0, 1], [0, 1], lw=.9, color="#999999", zorder=0)
ax1.set(xlim=(0, 1), ylim=(0, 1), xlabel="GAP, set marginal ($\\gamma=0$)",
        ylabel="GAP, calibrated ($\\gamma{=}0.25$)")
ax1.set_title(f"per target ({len(common)} items)", fontsize=9)
ax1.text(.04, .93, f"above line: {better}", fontsize=7.5, color="#2b5d8a")
ax1.text(.04, .86, f"below line: {worse}", fontsize=7.5, color="#999999")
delta = [b - c for b, c in zip(y, x)]
ax2.hist(delta, bins=45, color="#2b5d8a", alpha=.8)
ax2.axvline(0, lw=.9, color="#999999")
ax2.set(xlabel="$\\Delta$ GAP (calibrated $-$ set marginal)", ylabel="targets")
ax2.set_title(f"mean {sum(delta)/len(delta):+.4f}", fontsize=9)
for ax in (ax1, ax2):
    ax.tick_params(labelsize=7.5)
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(outdir / "fig_swords_per_target.pdf", bbox_inches="tight")
print("wrote", outdir / "fig_swords_per_target.pdf")

# ---- LaTeX fragment ------------------------------------------------------------
def esc(s):
    for ch, rep in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
                    ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
                    ("^", "\\textasciicircum{}")]:
        s = s.replace(ch, rep)
    return s

lines = [
"\\section{Qualitative analysis: which targets re-rank}",
"",
"\\label{sec:qualitative}",
"Table~\\ref{tab:qual} and Figure~\\ref{fig:pertarget} report where the ranking gain",
"comes from, at the level of individual SWORDS targets rather",
f"than the corpus mean. All numbers are the {a.mode} scoring mode on SWORDS dev",
f"({len(common)} targets), seed~42.",
"",
"\\begin{figure}[t]",
"\\centering",
"\\includegraphics[width=\\linewidth]{fig_swords_per_target.pdf}",
"\\caption{Per-target GAP for the calibrated objective against the released set",
f"marginal. The gain is broad rather than driven by outliers: {better} of {len(common)}",
f"targets improve, {worse} degrade, and the mean shift is {sum(delta)/len(delta):+.4f}.}}",
"\\label{fig:pertarget}",
"\\end{figure}",
"",
"\\paragraph{Top-1 flips.} Precision at one moves on few targets: on " + str(len(won)) + " the set",
"marginal's highest-ranked substitute is rejected by annotators while the calibrated model's is",
"accepted, and on " + str(len(lost)) + " the reverse. Table~\\ref{tab:qual} lists all of them, in both",
"directions, rather than a selection. This is consistent with precision at one being the one",
"ranking metric that does not separate the two objectives; the effect in Figure~\\ref{fig:pertarget}",
"is a broad shift in candidate ordering, not a handful of corrected top choices.",
"",
"\\begin{table*}[t]",
"\\centering\\small",
"\\begin{tabular}{p{0.46\\linewidth} p{0.30\\linewidth} rr}",
"\\toprule",
"Context (target in \\textbf{bold}) & Accepted substitutes & $\\gamma{=}0$ & $\\gamma{=}0.25$ \\\\",
"\\midrule",
]
direction_rows = {"gain": [], "loss": []}
for tid, direction in chosen:
    flat, word, off, raw = context_of(tid)
    # Escape BEFORE the ellipses go in, or esc() turns the \dots into literal text.
    left_words = " ".join(raw[:off].split()).split()
    right_words = " ".join(raw[off + len(word):].split()).split()
    left = esc(" ".join(left_words[-14:]))
    if len(left_words) > 14:
        left = "\\dots " + left
    right = esc(" ".join(right_words[:10]))
    if len(right_words) > 10:
        right = right + " \\dots"
    good = sorted([(s, h) for s, h in subs.get(tid, []) if h >= CONSERVATIVE],
                  key=lambda p: -p[1])[:5]
    shown = ", ".join(f"\\textit{{{esc(s)}}}" for s, _ in good) or "---"
    direction_rows[direction].append(
        f"{left} \\textbf{{{esc(word)}}} {right} & {shown} & "
        f"{per_target[ZHANG][tid]['gap']:.3f} & {per_target[HYBRID][tid]['gap']:.3f} \\\\")
for direction, heading in (("gain", "Calibrated ranks an accepted substitute first; set marginal does not"),
                           ("loss", "Set marginal ranks an accepted substitute first; calibrated does not")):
    if not direction_rows[direction]:
        continue
    lines.append("\\multicolumn{4}{l}{\\textit{" + heading + "}} \\\\")
    lines.append("\\addlinespace[2pt]")
    for row in direction_rows[direction]:
        lines.append(row); lines.append("\\addlinespace[2pt]")
    if direction == "gain":
        lines.append("\\midrule")
lines += [
"\\bottomrule",
"\\end{tabular}",
"\\caption{Every SWORDS dev target on which one objective ranks an accepted substitute",
"first and the other does not, listed exhaustively in both directions. The last two",
"columns are that target's GAP under each",
"objective. Accepted substitutes are those at least half the annotators marked \\textsc{true}.}",
"\\label{tab:qual}",
"\\end{table*}",
"",
]
frag = outdir / "appendix_qualitative.tex"
frag.write_text("\n".join(lines) + "\n")
print("wrote", frag, f"({len(chosen)} examples)")
