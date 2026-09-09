#!/usr/bin/env python3
"""Build contrastive splits that preserve gold identity, for the leak-free clean-gold corpus.

Why this wrapper exists
-----------------------
``build_contrastive_dataset.py`` writes only ``text, positives, negatives``.  Two columns it drops
are load-bearing downstream:

* ``gold_surface`` -- ``sequence_ncp_trainer.tokenize_concept_record`` runs with
  ``require_gold_for_concept=True``, and ``eval_concept_ppl_v3`` needs it to split the set marginal
  into GOLD and ALTERNATIVES.  Without it a contrastive arm cannot be scored on ALT NLL, which is
  the metric the comparison is about.
* ``row_id`` -- the paired bootstrap asserts identical held-out ``row_id`` sets across arms.

It also defaults to ``data/syn/youtube/``, the pre-July **leaked** splits.  This script reads
``youtube_clean_gold`` only.

Negatives are additionally filtered so that no negative is a positive or the gold surface, under
case folding.  A mined "negative" that is actually a valid substitute would train the model to
suppress a valid concept word -- the opposite of the objective -- and would make a null result
unattributable.

Output schema matches ``data/swords/concept_dev/contrastive_train.csv`` exactly:
``text, positives, negatives, gold_surface, row_id``
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from build_contrastive_dataset import get_hard_negatives


def parse_list(value) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def build_split(source_csv: Path, out_csv: Path, strategy: str, max_negatives: int,
                seed: int) -> dict:
    frame = pd.read_csv(source_csv)
    for column in ('text', 'context_syn', 'gold_surface', 'row_id'):
        if column not in frame.columns:
            raise ValueError(f'{source_csv} is missing required column {column!r}')

    rows, stats = [], {}
    n_with_negatives = n_negatives = n_dropped_overlap = 0
    for record in frame.to_dict('records'):
        positives = [str(p).strip() for p in parse_list(record['context_syn']) if str(p).strip()]
        gold = str(record['gold_surface']).strip()
        if not positives:
            continue
        mined = get_hard_negatives(positives, max_negatives=max_negatives,
                                   strategy=strategy, stats=stats)
        # A negative that is also a positive (or the gold surface) is a mining error, not a
        # negative.  Drop it rather than train the model to suppress a valid substitute.
        banned = {p.casefold() for p in positives} | {gold.casefold()}
        kept, seen = [], set()
        for word in mined:
            folded = str(word).strip().casefold()
            if not folded or folded in banned:
                n_dropped_overlap += 1
                continue
            if folded in seen:
                continue
            seen.add(folded)
            kept.append(str(word).strip())

        if kept:
            n_with_negatives += 1
            n_negatives += len(kept)
        rows.append({'text': record['text'], 'positives': str(positives),
                     'negatives': str(kept), 'gold_surface': gold,
                     'row_id': record['row_id']})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    total = len(rows)
    return {
        'source': str(source_csv), 'output': str(out_csv), 'strategy': strategy,
        'rows_in': int(len(frame)), 'rows_out': total,
        'rows_with_negatives': n_with_negatives,
        'negative_mining_coverage_pct': round(100 * n_with_negatives / total, 2) if total else 0.0,
        'negatives_total': n_negatives,
        'mean_negatives_per_covered_row': round(n_negatives / n_with_negatives, 2)
        if n_with_negatives else 0.0,
        'negatives_dropped_as_positive_or_gold': n_dropped_overlap,
        'sense_stats': {k: int(v) for k, v in sorted(stats.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--relation', choices=['syn', 'hyp'], default='syn')
    parser.add_argument('--gold_root', default=None,
                        help='defaults to data/<relation>/youtube_clean_gold')
    parser.add_argument('--output_dir', default=None,
                        help='defaults to the gold_root, so splits sit beside their source')
    parser.add_argument('--strategy', default='wrong_sense',
                        choices=['all', 'co_hyponym', 'wrong_sense', 'same_pos', 'none'],
                        help="default 'wrong_sense': co-hyponyms may still be valid substitutes "
                             'in context, which would make a null result unattributable')
    parser.add_argument('--max_negatives', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report_json', default=None)
    args = parser.parse_args()

    gold_root = Path(args.gold_root or f'data/{args.relation}/youtube_clean_gold')
    out_dir = Path(args.output_dir or gold_root)
    report = {'relation': args.relation, 'gold_root': str(gold_root), 'splits': {}}
    for split in ('train', 'val'):
        report['splits'][split] = build_split(
            gold_root / f'context_loss_{split}.csv',
            out_dir / f'contrastive_{split}.csv',
            args.strategy, args.max_negatives, args.seed)

    path = Path(args.report_json or out_dir / 'contrastive_report.json')
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'\nreport -> {path}')


if __name__ == '__main__':
    main()
