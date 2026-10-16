# Sealed holdout: creation report

Created 2026-10-16T21:06:46+00:00 at commit `5c2a2c659fd757a3bddcc14cf49eb2498253aabb` by `scripts/round2/build_sealed_holdout.py`. No model was loaded or scored.

**2000 targets** drawn (seed 20261016) from **6186** eligible CoInCo targets; 105378 substitutes.

## Pool construction and exclusions

| step | targets |
|---|---|
| dev: all | 10027 |
| dev: excluded, context of a SWORDS target (by CoInCo id) | 2278 |
| dev: excluded, in a used 800-subset | 800 |
| dev: excluded, sentence inside a SWORDS context (text) | 72 |
| dev: ineligible, needs >=1 accepted and >=1 rejected | 1044 |
| dev: pool before tokenizer check | 5833 |
| test: all | 5388 |
| test: excluded, context of a SWORDS target (by CoInCo id) | 4095 |
| test: excluded, in a used 800-subset | 800 |
| test: excluded, sentence inside a SWORDS context (text) | 42 |
| test: ineligible, needs >=1 accepted and >=1 rejected | 98 |
| test: pool before tokenizer check | 353 |

## Tokenizer eligibility (evaluator's own skip rules, both scorers)

| tokenizer | eligible | excluded |
|---|---|---|
| meta-llama/Llama-3.2-1B | 6186 | 0 |
| Qwen/Qwen3-1.7B-Base | 6186 | 0 |
| tiiuae/Falcon3-1B-Base | 6186 | 0 |

Only targets eligible for all three tokenizers are sampled.

## Strata (POS x lemma-frequency bucket)

Frequency = count of the target's (lemma, POS) among all 15,415 CoInCo targets; bucket edges are that distribution's quartiles (<=3, 4-8, 9-21, >=22).

| stratum | eligible | drawn |
|---|---|---|
| ADJ|1-3 | 203 | 66 |
| ADJ|22+ | 196 | 63 |
| ADJ|4-8 | 252 | 82 |
| ADJ|9-21 | 270 | 87 |
| ADV|1-3 | 60 | 19 |
| ADV|22+ | 349 | 113 |
| ADV|4-8 | 78 | 25 |
| ADV|9-21 | 144 | 47 |
| NOUN|1-3 | 581 | 188 |
| NOUN|22+ | 474 | 153 |
| NOUN|4-8 | 852 | 275 |
| NOUN|9-21 | 755 | 244 |
| VERB|1-3 | 246 | 80 |
| VERB|22+ | 874 | 283 |
| VERB|4-8 | 332 | 107 |
| VERB|9-21 | 520 | 168 |

## Pinned files

- `coinco_dev.json.gz` 02180be2bf9e3b528682c659c5b410f3baa130dbc19a29be8755a0aae0d9cc5f
- `coinco_test.json.gz` a6dc4b7708c38e947b6038a5fd2b6f0eab2381da61866afa75570953b3333bf4
- `coinco_dev_sub800.json.gz` 6679c2f22c1ee92469e6e24438d3b65edb86e1276ee19731af05d0d8f130962f
- `coinco_test_sub800.json.gz` 4400bf20e4ede042446743c08e536d3d4479a38515d92e3331e2ab34745c2bf2
- `swords-v1.1_dev.json.gz` bc3570786455a05f11fcb4fc7f54e117642d5fb5e98f19355028c5c7ee00c954
- `swords-v1.1_test.json.gz` f94b6b73c52fd4bd549180bc119d44e9f9097ec78240a8b42363add158e58480
- evaluation file `results/round2/sealed_data/coinco_sealed.json.gz` ea2dd771cb45a89691461ec6e23fe84e116d469098f7275fe5f03a8046ea2088 (derived, not committed)
