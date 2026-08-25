# bm-semlex (Ivan, github.com/ioana-ivan/bm-semlex)

Pinned commit: `2677b1ecdb28d9bd427da51a0b957d4dbe770f14`.

`curated_200.tsv` — TSV, no header, 200 human-curated rows:
`target <TAB> synonym <TAB> distractor <TAB> token_index <TAB> sentence`

Built from SemCor + WordNet. Each row is a minimal pair: in the given sentence the `synonym`
is a valid in-context substitute for `target` and the `distractor` (another WordNet synonym
of `target`, from a different sense) is not.

`wordnet_nouns_10000.txt` — WordNet noun list used there for random-negative sampling.

Use as PRIOR ART and an optional 200-pair minimal-pair sanity check, not a primary benchmark.
The upstream repo reports the same style of LM-perplexity-on-synonym test for OLMo-1B/7B and Amber.

`curated_200.tsv` SHA-256:
`87cb29fe77b27e9ed5f64c08691209ff464399962013ed673fc35d3b107ea5ab`.
