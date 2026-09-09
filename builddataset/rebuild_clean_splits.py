#!/usr/bin/env python3
"""
Rebuild leak-free, context-grouped train/val splits (July 2026 validity fix).

PROBLEM (measured on data/{syn,hyp}/youtube):
    syn context_loss_val  ∩ hyp context_loss_train : 220 / 222 unique contexts
    hyp context_loss_val  ∩ syn context_loss_train :   6 / 309
    plus prefix-level leakage between val contexts and train txt lines.

The syn and hyp datasets were split into train/val independently, but both were
derived from the SAME underlying sentences. Any model trained on hyp (or merged
syn+hyp) data has therefore seen ~99% of the synonym validation contexts, which
invalidates concept-PPL numbers computed on syn val for those models.

FIX:
  1. Pool every unique context string across syn/hyp x train/val
     (context_loss and dict_loss CSVs share identical text columns row-for-row).
  2. Group contexts that are word-boundary prefixes of one another — different
     concept slots of the same source sentence produce nested prefixes, so
     grouping guarantees they travel to the same side of the split.
     "<mask> ..."-style rows are attached to their source sentence by suffix
     matching against the vanilla text lines.
  3. Split at the GROUP level (seeded, --val_frac of rows).
  4. Re-emit all CSVs (context_loss + dict_loss, syn + hyp) routed by group.
  5. Re-emit the txt files (context_syn_*.txt, vanilla_*.txt): each line is
     routed to the side of the longest context group that prefixes it; lines
     matching no group keep their original side.
  6. Audit: hard-fails (exit 1) if any val context appears in any train
     artifact after the rebuild.

Outputs (default):
    data/syn/youtube_clean/{context_loss,dict_loss}_{train,val}.csv
    data/syn/youtube_clean/{context_syn,vanilla}_{train,val}.txt
    data/hyp/youtube_clean/...                      (same layout)
    data/youtube_clean_split_report.json

Stdlib-only (csv/json/random) so it runs anywhere, incl. bare Colab.
"""

import argparse
import bisect
import csv
import json
import os
import random
import sys
from collections import defaultdict

# Characters that may legally follow a context prefix inside its source sentence.
BOUNDARY_CHARS = set(" \t\n'’“”\"“”,.!?;:()-")

MASK_TOKEN = "<mask>"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def load_txt(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


# ── Grouping ─────────────────────────────────────────────────────────────────

def is_boundary_prefix(base, s):
    """True if `base` is a prefix of `s` ending at a word boundary."""
    if not s.startswith(base):
        return False
    if len(s) == len(base):
        return True
    return s[len(base)] in BOUNDARY_CHARS or base.endswith((" ", MASK_TOKEN))


def build_groups(contexts):
    """
    Group contexts sharing a word-boundary prefix.

    Sorted lexicographically, all strings sharing prefix P form a contiguous
    block starting at P itself, so a single pass with a running group base
    captures every prefix chain (conservative: over-grouping is leak-safe,
    under-grouping is not).

    Returns (group_of: context -> gid, n_groups).
    """
    uniq = sorted(set(contexts))
    group_of = {}
    gid = -1
    base = None
    for s in uniq:
        if base is not None and is_boundary_prefix(base, s):
            group_of[s] = gid
        else:
            gid += 1
            base = s
            group_of[s] = gid
    return group_of, gid + 1


def attach_mask_rows(group_of, all_lines):
    """
    "<mask> tail..." contexts have the concept slot at position 0, so prefix
    grouping cannot link them to their source sentence. Attach each to the
    group of a vanilla/augmented line ending with the same tail. If the source
    line itself matches no prefix-context group, bind the LINE to the mask
    context's group instead (returned as line_overrides), so line and context
    always travel to the same side of the split.
    """
    line_group = {}
    sorted_ctx = sorted(group_of)

    def longest_prefix_group(line):
        # All prefixes of `line` sort <= line; walk candidates via bisect.
        i = bisect.bisect_right(sorted_ctx, line)
        best = None
        for j in range(i - 1, max(i - 200, -1), -1):  # bounded back-walk
            c = sorted_ctx[j]
            if is_boundary_prefix(c, line):
                best = c
                break
            if line[: len(c)] > c and not line.startswith(c[:1]):
                break
        return group_of.get(best) if best else None

    for line in all_lines:
        g = longest_prefix_group(line)
        if g is not None:
            line_group[line] = g

    # A mask tail may appear in SEVERAL lines (near-duplicate comments); every
    # such line's group must be merged with the mask context's group, or the
    # tail can straddle the split. Collect merge sets and resolve via union-find.
    reassigned = 0
    line_overrides = {}  # line text -> gid (for lines with no prefix-context group)
    merges = []          # sets of gids that must end up on the same side
    for ctx in list(group_of):
        if not ctx.startswith(MASK_TOKEN):
            continue
        tail = ctx[len(MASK_TOKEN):].strip()
        if not tail:
            continue
        gids = {group_of[ctx]}
        matched = False
        for line in all_lines:
            l = line.strip()
            if l.endswith(tail) or (len(tail) >= 10 and tail in l):
                matched = True
                if line in line_group:
                    gids.add(line_group[line])
                else:
                    line_overrides[line.strip()] = group_of[ctx]
        if matched:
            reassigned += 1
        if len(gids) > 1:
            merges.append(gids)

    # Union-find over group ids.
    parent = {}

    def find(g):
        parent.setdefault(g, g)
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for gids in merges:
        it = iter(gids)
        root = find(next(it))
        for g in it:
            parent[find(g)] = root

    # Canonicalize all assignments to union roots.
    for c in group_of:
        group_of[c] = find(group_of[c])
    line_overrides = {l: find(g) for l, g in line_overrides.items()}
    return reassigned, line_overrides


# ── Splitting ────────────────────────────────────────────────────────────────

def split_groups(group_row_counts, val_frac, seed):
    gids = sorted(group_row_counts)
    rng = random.Random(seed)
    rng.shuffle(gids)
    total = sum(group_row_counts.values())
    val_groups, acc = set(), 0
    for g in gids:
        if acc / total >= val_frac:
            break
        val_groups.add(g)
        acc += group_row_counts[g]
    return val_groups


# ── Line routing (txt files) ─────────────────────────────────────────────────

def route_line(line, ctx_set, max_ctx_len, group_of):
    """Longest context in ctx_set that boundary-prefixes `line` -> its group."""
    limit = min(len(line), max_ctx_len)
    for L in range(limit, 0, -1):
        cand = line[:L]
        if cand in ctx_set and is_boundary_prefix(cand, line):
            return group_of[cand]
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Rebuild leak-free context-grouped splits")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--in_subdir", default="youtube")
    ap.add_argument("--out_subdir", default="youtube_clean")
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_json", default=None,
                    help="default: <data_root>/<out_subdir>_split_report.json")
    args = ap.parse_args()

    root = args.data_root
    report = {"seed": args.seed, "val_frac": args.val_frac, "before": {}, "after": {}}

    # 1. Load all CSVs -------------------------------------------------------
    csv_specs = []  # (source, family, split, fieldnames, rows)
    for source in ("syn", "hyp"):
        for family in ("context_loss", "dict_loss"):
            for split in ("train", "val"):
                path = os.path.join(root, source, args.in_subdir, f"{family}_{split}.csv")
                if not os.path.exists(path):
                    print(f"  (missing, skipped: {path})")
                    continue
                fieldnames, rows = load_csv(path)
                csv_specs.append((source, family, split, fieldnames, rows))

    # 2. Pre-rebuild audit ----------------------------------------------------
    def ctx_set_of(source, family, split):
        for s, f, sp, _, rows in csv_specs:
            if (s, f, sp) == (source, family, split):
                return {r["text"].strip() for r in rows}
        return set()

    sv = ctx_set_of("syn", "context_loss", "val")
    hv = ctx_set_of("hyp", "context_loss", "val")
    st = ctx_set_of("syn", "context_loss", "train")
    ht = ctx_set_of("hyp", "context_loss", "train")
    report["before"] = {
        "syn_val ∩ hyp_train": f"{len(sv & ht)} / {len(sv)}",
        "syn_val ∩ syn_train": f"{len(sv & st)} / {len(sv)}",
        "hyp_val ∩ syn_train": f"{len(hv & st)} / {len(hv)}",
        "hyp_val ∩ hyp_train": f"{len(hv & ht)} / {len(hv)}",
    }
    print("Pre-rebuild leakage audit (exact context overlap):")
    for k, v in report["before"].items():
        print(f"  {k}: {v}")

    # 3. Build groups over ALL contexts ---------------------------------------
    all_contexts = []
    for _, family, _, _, rows in csv_specs:
        if family == "context_loss":  # dict_loss texts are identical row-for-row
            all_contexts.extend(r["text"].strip() for r in rows)
    group_of, n_groups = build_groups(all_contexts)
    print(f"\nGrouped {len(set(all_contexts))} unique contexts into {n_groups} groups.")

    # Attach <mask> rows to their source sentence via txt tails.
    txt_specs = []  # (source, name, split, lines)
    for source in ("syn", "hyp"):
        for name in ("context_syn", "vanilla"):
            for split in ("train", "val"):
                path = os.path.join(root, source, args.in_subdir, f"{name}_{split}.txt")
                lines = load_txt(path)
                if lines:
                    txt_specs.append((source, name, split, lines))
    all_lines = [l for _, _, _, lines in txt_specs for l in lines]
    n_mask_fixed, line_overrides = attach_mask_rows(group_of, all_lines)
    n_mask_total = sum(1 for c in group_of if c.startswith(MASK_TOKEN))
    print(f"<mask> contexts attached to source-sentence groups: {n_mask_fixed}/{n_mask_total}")

    # Group size distribution sanity check.
    group_members = defaultdict(int)
    for c in set(all_contexts):
        group_members[group_of[c]] += 1
    biggest = max(group_members.values()) if group_members else 0
    if biggest > 0.05 * len(set(all_contexts)):
        print(f"NOTE: largest group holds {biggest} contexts (>5%). Expected cause: chains of "
              f"very short contexts ('I' -> 'I have' -> ...). Over-grouping is leak-safe; it only "
              f"means these contexts travel to the same split side as one unit.")
    report["n_groups"] = n_groups
    report["largest_group_contexts"] = biggest

    # 4. Split at group level (weighted by context_loss row counts) -----------
    group_row_counts = defaultdict(int)
    for _, family, _, _, rows in csv_specs:
        if family == "context_loss":
            for r in rows:
                group_row_counts[group_of[r["text"].strip()]] += 1
    val_groups = split_groups(group_row_counts, args.val_frac, args.seed)
    print(f"Val groups: {len(val_groups)}/{n_groups}")

    def side_of(ctx):
        return "val" if group_of[ctx] in val_groups else "train"

    # 5. Emit CSVs -------------------------------------------------------------
    out_rows = defaultdict(list)  # (source, family, split) -> rows
    out_fields = {}
    for source, family, _, fieldnames, rows in csv_specs:
        out_fields[(source, family)] = fieldnames
        for r in rows:
            out_rows[(source, family, side_of(r["text"].strip()))].append(r)

    for (source, family, split), rows in sorted(out_rows.items()):
        out_dir = os.path.join(root, source, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{family}_{split}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=out_fields[(source, family)])
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {len(rows):5d} rows -> {out_path}")
        report["after"][f"{source}/{family}_{split}"] = len(rows)

    # 6. Emit txt files ---------------------------------------------------------
    ctx_pool = set(group_of)
    max_ctx_len = max((len(c) for c in ctx_pool), default=0)

    # Unmatched lines keep their original side — but identical text must not
    # end up on both sides. Any unmatched text seen in val AND train goes to train.
    unmatched_sides = defaultdict(set)
    line_route_cache = {}
    for source, name, orig_split, lines in txt_specs:
        for line in lines:
            key = line.strip()
            if key not in line_route_cache:
                g = line_overrides.get(key)
                if g is None:
                    g = route_line(key, ctx_pool, max_ctx_len, group_of)
                line_route_cache[key] = g
            if line_route_cache[key] is None:
                unmatched_sides[key].add(orig_split)

    txt_out = defaultdict(list)  # (source, name, split) -> lines
    n_routed, n_kept = 0, 0
    for source, name, orig_split, lines in txt_specs:
        for line in lines:
            g = line_route_cache[line.strip()]
            if g is None:
                sides = unmatched_sides[line.strip()]
                dest = "train" if len(sides) > 1 else orig_split
                n_kept += 1
            else:
                dest = "val" if g in val_groups else "train"
                n_routed += 1
            txt_out[(source, name, dest)].append(line)

    for (source, name, split), lines in sorted(txt_out.items()):
        out_dir = os.path.join(root, source, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}_{split}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  wrote {len(lines):5d} lines -> {out_path}")
        report["after"][f"{source}/{name}_{split}.txt"] = len(lines)
    print(f"txt lines routed by context group: {n_routed}; kept original side: {n_kept}")

    # 7. Post-rebuild audit — hard-fail on any residual leak --------------------
    def clean_ctx(source, family, split):
        return {r["text"].strip() for r in out_rows[(source, family, split)]}

    val_ctx = clean_ctx("syn", "context_loss", "val") | clean_ctx("hyp", "context_loss", "val")
    train_ctx = clean_ctx("syn", "context_loss", "train") | clean_ctx("hyp", "context_loss", "train") \
        | clean_ctx("syn", "dict_loss", "train") | clean_ctx("hyp", "dict_loss", "train")

    leaks = {}
    leaks["val_ctx_in_train_ctx"] = len(val_ctx & train_ctx)

    # val contexts that prefix a clean train txt line (model would train on them)
    n_prefix_leak = 0
    n_mask_tail_leak = 0
    train_lines = [l.strip() for (s, n, sp), ls in txt_out.items() if sp == "train" for l in ls]
    for c in val_ctx:
        if c.startswith(MASK_TOKEN):
            tail = c[len(MASK_TOKEN):].strip()
            if len(tail) >= 10 and any(tail in l for l in train_lines):
                n_mask_tail_leak += 1
            continue
        for l in train_lines:
            if is_boundary_prefix(c, l):
                n_prefix_leak += 1
                break
    leaks["val_ctx_prefixing_train_txt"] = n_prefix_leak
    leaks["mask_val_ctx_tail_in_train_txt"] = n_mask_tail_leak

    # identical txt line on both sides
    val_lines_set = {l.strip() for (s, n, sp), ls in txt_out.items() if sp == "val" for l in ls}
    leaks["txt_line_on_both_sides"] = len(val_lines_set & set(train_lines))

    report["after"]["leak_audit"] = leaks
    print("\nPost-rebuild leakage audit:")
    for k, v in leaks.items():
        print(f"  {k}: {v}")

    report_path = args.report_json or os.path.join(root, f"{args.out_subdir}_split_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {report_path}")

    if any(v > 0 for v in leaks.values()):
        print("FAIL: residual leakage detected — outputs written but DO NOT use.")
        sys.exit(1)
    print("OK: no residual leakage. Clean splits ready.")


if __name__ == "__main__":
    main()
