# SWORDS v1.1 — Stanford Word Substitution Benchmark

Source: https://github.com/p-lambda/swords (`assets/parsed/`)
Pinned commit: `04ca75370d0ce098a7f4db68240fc8e79a4f7b3b`
Paper: Lee, Donahue, Jia, Iyabor, Liang — *Swords: A Benchmark for Lexical Substitution with
Improved Data Coverage and Quality* (NAACL 2021).

| file | contexts/targets | substitutes | human labels |
|---|---|---|---|
| `swords-v1.1_dev.json.gz`  | 370 | 22,978 | 3 or 10 raters each |
| `swords-v1.1_test.json.gz` | 762 | 45,705 | 3 or 10 raters each |

JSON keys: `contexts`, `targets`, `substitutes`, `substitutes_lemmatized`, `substitute_labels`.
Labels are `TRUE` / `FALSE` / `UNSURE` per rater. Acceptability = fraction `TRUE`.

Test-split label distribution: mean acceptability 0.122; 9.7% of substitutes reach the
conservative threshold (>= 0.5), 36.5% the lenient one (>= 0.1).

POS coverage (test): NOUN 280, VERB 294, ADJ 125, ADV 63. Note this project's own data is
**nouns only**, so the non-noun strata are out-of-distribution and should be reported separately.

Evaluate with `transformers/examples/pytorch/language-modeling/eval_swords.py`.

Integrity:

- dev SHA-256: `bc3570786455a05f11fcb4fc7f54e117642d5fb5e98f19355028c5c7ee00c954`
- test SHA-256: `f94b6b73c52fd4bd549180bc119d44e9f9097ec78240a8b42363add158e58480`
- license: CC-BY-3.0-US, as declared by the upstream repository
