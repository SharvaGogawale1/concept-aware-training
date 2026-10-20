# Sealed holdout: creation report

Created 2026-10-20T13:14:06+00:00 at commit `a04067d52460171d108f27add1aadd78e1ed16ff` by `scripts/round2/build_sealed_holdout.py`. No model was loaded or scored.

**3370 targets in 784 sentences**, sampling: none: every eligible target (seed 20261016 applies only to a draw), from **3370** eligible CoInCo targets in 784 sentences (floor 400 sentences); 176865 substitutes.

Version 2. Exclusion rule: a target is excluded if it is in coinco_dev_sub800 or coinco_test_sub800, if its sentence (CoInCo context) holds any target of either subset, or if its sentence is a SWORDS dev/test context (by CoInCo id or by text).

Targets per sentence in the draw: 1: 129 sentences, 2: 126 sentences, 3: 125 sentences, 4: 96 sentences, 5: 86 sentences, 6: 65 sentences, 7: 56 sentences, 8: 31 sentences, 9: 19 sentences, 10: 15 sentences, 11: 12 sentences, 12: 10 sentences, 13: 6 sentences, 14: 3 sentences, 15: 3 sentences, 16: 2 sentences.

Supersedes `sealed/superseded_v1/sealed_holdout_ids.json` (sha256 b2fdab91dfa7973ebf445ae10a6c8612c413945e12c7d4e672a4c8ae9444d5ef, 2000 targets): version 1 excluded the used 800-subset targets but not their sentences; 912 of its 2000 targets shared a sentence with an already-scored target.  No model was scored on it. 1088 of its targets are drawn again here.

## Pool construction and exclusions

| step | targets |
|---|---|
| dev: all | 10027 |
| dev: excluded, context of a SWORDS target (by CoInCo id) | 1221 |
| dev: excluded, in a used 800-subset | 800 |
| dev: excluded, sentence inside a SWORDS context (text) | 22 |
| dev: excluded, shares a sentence with a used 800-subset target | 4215 |
| dev: ineligible, needs >=1 accepted and >=1 rejected | 533 |
| dev: pool before tokenizer check | 3236 |
| test: all | 5388 |
| test: excluded, context of a SWORDS target (by CoInCo id) | 1353 |
| test: excluded, in a used 800-subset | 800 |
| test: excluded, sentence inside a SWORDS context (text) | 12 |
| test: excluded, shares a sentence with a used 800-subset target | 3050 |
| test: ineligible, needs >=1 accepted and >=1 rejected | 39 |
| test: pool before tokenizer check | 134 |

## Tokenizer eligibility (evaluator's own skip rules, both scorers)

| tokenizer | eligible | excluded |
|---|---|---|
| meta-llama/Llama-3.2-1B | 3370 | 0 |
| Qwen/Qwen3-1.7B-Base | 3370 | 0 |
| tiiuae/Falcon3-1B-Base | 3370 | 0 |

Only targets eligible for all three tokenizers are sampled.

## Strata (POS x lemma-frequency bucket; recorded, not sampled when every target is taken)

Frequency = count of the target's (lemma, POS) among all 15,415 CoInCo targets; bucket edges are that distribution's quartiles (<=3, 4-8, 9-21, >=22).

| stratum | eligible | drawn |
|---|---|---|
| ADJ|1-3 | 114 | 114 |
| ADJ|22+ | 93 | 93 |
| ADJ|4-8 | 142 | 142 |
| ADJ|9-21 | 134 | 134 |
| ADV|1-3 | 31 | 31 |
| ADV|22+ | 200 | 200 |
| ADV|4-8 | 42 | 42 |
| ADV|9-21 | 82 | 82 |
| NOUN|1-3 | 303 | 303 |
| NOUN|22+ | 253 | 253 |
| NOUN|4-8 | 445 | 445 |
| NOUN|9-21 | 411 | 411 |
| VERB|1-3 | 138 | 138 |
| VERB|22+ | 508 | 508 |
| VERB|4-8 | 186 | 186 |
| VERB|9-21 | 288 | 288 |

## Pinned files

- `coinco_dev.json.gz` 02180be2bf9e3b528682c659c5b410f3baa130dbc19a29be8755a0aae0d9cc5f
- `coinco_test.json.gz` a6dc4b7708c38e947b6038a5fd2b6f0eab2381da61866afa75570953b3333bf4
- `coinco_dev_sub800.json.gz` 6679c2f22c1ee92469e6e24438d3b65edb86e1276ee19731af05d0d8f130962f
- `coinco_test_sub800.json.gz` 4400bf20e4ede042446743c08e536d3d4479a38515d92e3331e2ab34745c2bf2
- `swords-v1.1_dev.json.gz` bc3570786455a05f11fcb4fc7f54e117642d5fb5e98f19355028c5c7ee00c954
- `swords-v1.1_test.json.gz` f94b6b73c52fd4bd549180bc119d44e9f9097ec78240a8b42363add158e58480
- evaluation file `results/round2/sealed_data/coinco_sealed_v2.json.gz` db8e2973f0bd1daf95d1d85fa713158768f03e8e35ae0d5a8ff60cc2d71348a2 (derived, not committed)
