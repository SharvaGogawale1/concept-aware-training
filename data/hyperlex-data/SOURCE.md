# HyperLex

Source: https://github.com/cambridgeltl/hyperlex (Vulić, Gerz, Kiela, Hill, Korhonen —
*HyperLex: A Large-Scale Evaluation of Graded Lexical Entailment*, CL 2017).
Pinned commit: `ccb416191a46e37f7104a530e7c23f260e066e8c`.

2,616 word pairs (2,163 noun, 453 verb) rated 0–6 on "To what degree is X a type of Y?".

`hyperlex-all.txt` columns: `WORD1 WORD2 POS TYPE AVG_SCORE AVG_SCORE_0_10 STD SCORES..`

Relation types and mean ratings (all pairs):

| type | n | mean |
|---|---|---|
| hyp-1 … hyp-4 | 384/290/289/242 | 4.72 – 5.00 |
| syn | 194 | 4.10 |
| r-hyp-1 … r-hyp-4 | 98/73/75/50 | 2.85 – 1.71 |
| cohyp | 292 | 2.13 |
| mero | 241 | 1.89 |
| ant | 98 | 0.88 |
| no-rel | 290 | 0.51 |

`splits/random/` is the standard split; `splits/lexical/` guarantees no word overlap between
train and test, so it is the harder and more honest one for any method that touches training data.
`r-hyp` pairs are the `hyp` pairs reversed — the directionality contrast.

Evaluate with `transformers/examples/pytorch/language-modeling/eval_hyperlex.py`.

The upstream `hyperlex-data.zip` SHA-256 is
`27ef061e2f711d8e8a53b9fe204eaf06c596ef53d95bbed81ae52c6eebe5713b`.
