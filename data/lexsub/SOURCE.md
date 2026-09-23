# Lexical-substitution benchmarks beyond SWORDS

Nothing in this directory is authored by us and nothing in it is committed.

`raw/` holds the pinned sources, fetched by `builddataset/verify_task14_data.py
--download_missing` against the URLs and SHA-256 digests in
`data/external_benchmarks_manifest.json`:

- `coinco_{dev,test}.json.gz` -- CoInCo (Kremer et al., 2014) as parsed by the
  SWORDS repository at the commit this repo pins.
- `task10data.tar.gz` -- SemEval-2007 Task 10 (McCarthy & Navigli, 2007), the
  official release with test data and gold.
- `lst.gold.candidates` -- Melamud et al.'s per-lexelt candidate pools for it.

The three evaluation files beside this note are derived from them by
`builddataset/build_lexsub_benchmarks.py`, which pins their digests too:

- `semeval07_test.json.gz` -- all 1,703 gold-bearing test instances.
- `coinco_{dev,test}_sub800.json.gz` -- a fixed 800-target subset of each split,
  drawn once with seed 0 before any model was scored.

Protocol: `coinco_dev_sub800` is scored beside SWORDS dev during development;
`semeval07_test` and `coinco_test_sub800` are scored only in the locked test
cell, together with SWORDS test.
