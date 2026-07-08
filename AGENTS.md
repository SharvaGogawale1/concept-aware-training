# AGENTS.md — Concept-Aware Training

## Project Overview

Collaborative NLP research with **Sharva Gogawale** (Tel Aviv) and **Chen Shani** (Stanford), building on Laya's original concept-aware LM training codebase. Goal: train LLMs to assign probability to *any* valid concept in a synonym/hypernym set at concept slots in text, not just the single gold token. Targeting EMNLP/ACL.

## Repository Structure

```
concept-aware-training/
├── data/
│   ├── syn/youtube/          # Synonym datasets (LEAKED splits — superseded, see below)
│   ├── syn/youtube_clean/    # Leak-free splits (July 2026) — USE THESE
│   ├── hyp/youtube/          # Hypernym datasets (LEAKED splits — superseded)
│   ├── hyp/youtube_clean/    # Leak-free splits (July 2026) — USE THESE
│   └── contrastive/          # youtube/ = June (leaked); youtube_clean*/ = revised round
├── build_contrastive_dataset.py   # Hard-negative mining; wrong-sense FIXED July 2026; --strategy incl. 'none'
├── rebuild_clean_splits.py        # July 2026: context-grouped leak-free resplit + audit (hard-fails on leak)
├── concept_aware_training_colab.ipynb   # Tasks 1–4
├── research_tasks_5-8.ipynb            # Tasks 5–8 June round — results SUPERSEDED (leak + broken ablation)
├── research_tasks_5_8_revised.ipynb    # REVISED round (July 2026) — canonical going forward
├── README.md
└── transformers/examples/pytorch/language-modeling/
    ├── run_clm.py                    # Baseline CLM training
    ├── run_clm_syn_custom_loss.py    # Original synonym NCP training
    ├── run_clm_hyp_custom_loss.py    # Original hypernym NCP training
    ├── custom_trainer.py             # Original NCP trainer (uses torch.no_grad — broken grads)
    ├── hierarchical_trainer.py       # Hypernym NCP trainer
    ├── eval_ntp_baselines.py         # Task 1: vanilla NTP eval across checkpoints
    ├── eval_concept_ppl.py           # Task 5 v1 — DEPRECATED (circular metric, eval-time resize)
    ├── eval_concept_ppl_v2.py        # Task 5R: canonical tokenizer, in-context multi-token scoring, bootstrap CIs
    ├── differentiable_ncp_trainer.py # Task 2: differentiable NCP via logsumexp
    ├── run_clm_differentiable_ncp.py # Task 2: training script
    ├── contrastive_trainer.py        # Task 4: InfoNCE contrastive trainer
    ├── run_clm_contrastive.py        # Task 4 & 6: training script
    └── run_downstream_eval.py        # Task 7: fine-tune on SNLI / SPAM, report accuracy + F1
```

## Key Data Format

- `context_loss_train.csv`: columns `text` (context prefix) and `context_syn` (Python list of valid concept words, stored as string)
- `context_syn_train.txt`: augmented full sentences (for standard CLM)
- `vanilla_train.txt`: plain text without augmentation (for baseline eval)
- Tokenized sequence structure: `[BOS=128000] + padded_128_tokens + [EOS=128001]`
- The lookup key used in all trainers: `str([BOS] + padded_content + [EOS])`

## Four Implemented Tasks

### Task 1 — NTP Baseline Evaluation (`eval_ntp_baselines.py`)
Evaluates multiple checkpoints with **vanilla CLM loss** on plain validation text. Separates eval protocol from training objective. Takes `--checkpoints` (list), `--validation_file`, `--results_json`.

### Task 2 — Differentiable NCP (`differentiable_ncp_trainer.py` + `run_clm_differentiable_ncp.py`)
Fixes the core bug in `custom_trainer.py`: original runs N `torch.no_grad()` forward passes for concept scoring, killing gradient flow. Fix: extract concept-slot logits from the **existing base forward pass** and apply `torch.logsumexp` — zero extra compute, fully differentiable. Key arg: `--ncp_alpha` (default 1.0).

### Task 3 — Hard-Negative Dataset Builder (`build_contrastive_dataset.py`)
Mines WordNet hard negatives per concept set via three strategies: (1) co-hyponyms, (2) wrong-sense distractors, (3) same-POS fallback. Outputs `contrastive_train.csv` with columns `text`, `positives`, `negatives`. Run from repo root. Requires `pip install nltk` + wordnet download.

### Task 4 — Contrastive Training (`contrastive_trainer.py` + `run_clm_contrastive.py`)
InfoNCE-style loss: `L = L_CLM + alpha * NCP_positives + beta * InfoNCE`. InfoNCE per slot: `logsumexp([log_p_pos, log_p_neg]) - log_p_pos`. All scores from the existing forward pass. Key args: `--ncp_alpha` (default 0.5), `--contrast_beta` (default 1.0). Requires Task 3 output.

## REVISED ROUND (July 2026) — Validity Fixes: READ BEFORE USING ANY RESULTS

A code/data audit (2026-07-06) invalidated several June results. **All numbers from
`research_tasks_5_8.ipynb` and the June "Observed Results" table below are superseded**
by `research_tasks_5_8_revised.ipynb`. What was wrong:

1. **Train/val leakage**: 220/222 unique syn-val contexts appear in hyp-train (the syn and hyp
   datasets were split independently but derive from the same sentences). Any model trained on
   hyp or merged data saw ~99% of the syn val set. The famous merged-contrastive concept PPL of
   1.37 was memorization; the merged-vs-syn-only gap (1.37 vs 1443.8) was the leak, not evidence
   that hypernym data helps. Fixed by `rebuild_clean_splits.py` (context-grouped resplit; also
   covers dict_loss CSVs — identical text columns — the txt files, and "<mask>"-style rows via
   tail matching; hard-fails if any residual overlap remains).
2. **Wrong-sense mining was structurally empty** (0% coverage): old code treated every synset of
   every positive as "intended", so non-intended senses could not exist. The June "wrong-sense
   only" ablation was actually a NO-NEGATIVES run. Fixed via synonym-intersection sense
   disambiguation (`_intended_synsets`: synsets containing ≥2 positives; first-sense fallback)
   plus a hierarchy filter (ancestors/descendants of the intended sense are not "wrong senses").
   Real coverage now ~86–88%. `--strategy none` added as the explicit no-negatives control.
3. **v1 concept eval was circular**: it scored bare-word token ids (no leading space) — exactly
   the convention the trainers optimize, which CLM never trained on — and it resized embeddings
   at eval time (random row in a tied softmax) with per-checkpoint tokenizers.
   `eval_concept_ppl_v2.py` fixes all of it: one canonical tokenizer (`--tokenizer_path`), never
   resizes, scores candidates as in-context continuations (multi-token supported), reports mean
   NLL + 95% bootstrap CI, and separates eval-slot coverage from negative-mining coverage.

Hard rules going forward:
- Train and evaluate ONLY on `data/*/youtube_clean/`. Never mix old and clean splits.
- Never evaluate pre-July checkpoints on the clean val sets (they trained on old train sides
  that intersect the clean val side). The revised notebook retrains everything.
- Concept-PPL numbers from v1 and v2 are NOT comparable (different tokenization convention).
- Always report negative-mining coverage (builder) and eval-slot coverage (v2) as separate stats.

## Research Tasks 5–8 — Chen's Priority Experiments

These tasks follow Chen's feedback after the initial round of results (Tasks 1–4). Priority order matches Chen's recommendation.

### Observed Results (Tasks 1–4 Round) — SUPERSEDED (leaked splits + circular v1 metric; see REVISED ROUND)

| Model | Vanilla PPL | Vanilla Acc |
|---|---|---|
| Standard CLM | 21.84 | 60.1% |
| Hypernym NCP | 42.91 | 39.5% |
| Contrastive (α=0.5, β=1.0) | 43.65 | **45.9%** |
| Synonym NCP | 48.55 | 37.4% |
| Differentiable NCP (α=1.0) | 54.74 | 43.9% |

Key finding: all concept-trained models underperform CLM on NTP PPL, but the contrastive model achieves the best accuracy among concept-trained variants (+6pp over original NCP). The NTP↓ / accuracy↑ tradeoff is expected — concept training spreads probability mass across a valid set, which by definition hurts single-token NTP.

### Task 5 — Dual Evaluation: Concept PPL + NTP PPL (`eval_concept_ppl.py`)

Chen explicitly requested both metrics together. Three metrics are reported side-by-side:
- **NTP PPL / Acc**: standard CLM loss on `vanilla_val.txt` (already in Task 1)
- **Set-marginal PPL**: `exp(mean(-log Σ_{c ∈ C} p(c|context)))` on `context_loss_val.csv`

The set-marginal PPL is the strongest intrinsic metric for concept coverage — it directly measures whether the model assigns probability mass to any member of the valid concept set, regardless of which surface form appears in the gold text.

Script: `eval_concept_ppl.py`. Args: `--checkpoints`, `--concept_csv` (context_loss_val.csv), `--vanilla_val`, `--results_json`.

### Task 6 — Syn-Only Fair Comparison (highest priority: controls confound)

**Why this is critical**: The contrastive model was trained on the merged syn+hyp dataset (2,241 examples) while original NCP models used syn-only (~1,828 examples). Before attributing the +6pp accuracy gain to the contrastive objective, we must rule out data size/domain mixture as the cause.

Steps:
1. `python build_contrastive_dataset.py --source syn --output_dir data/contrastive/youtube_syn_only` — produces syn-only contrastive CSV
2. Train: `run_clm_contrastive.py --train_file data/contrastive/youtube_syn_only/syn_contrastive_train.csv`
3. Evaluate with `eval_concept_ppl.py` (both concept PPL and NTP PPL)
4. Compare: Contrastive (syn-only) vs. Contrastive (merged) vs. Syn NCP (original)

If the syn-only contrastive model still beats syn NCP, the gain is attributable to the objective; if not, the gain was a data artifact.

### Task 7 — Downstream Evaluation: SNLI + SPAM (`run_downstream_eval.py`)

Highest priority per Chen. Tests whether the accuracy gain from contrastive training transfers to NLU tasks.

- **SNLI** (3-class NLI: entailment/neutral/contradiction). Hypothesis: hypernym NCP and contrastive training should improve entailment detection since teaching `cat → animal` is precisely the is-a relationship NLI tests.
- **SPAM** (`talby/spamassassin`, config=`"text"` — SpamAssassin public mail corpus). Binary ham/spam classification. Only has a `train` split (10.7k rows); `run_downstream_eval.py` auto-creates an 80/20 train/val split. Fields: `label` (ClassLabel: 0=ham, 1=spam), `text`.

Protocol:
1. Linear probe first (`--freeze_base`): freeze LM parameters, train only a classification head. Measures pre-training representation quality in isolation.
2. Full fine-tune (`--no-freeze`): train all parameters. Measures best achievable downstream performance.

Script: `run_downstream_eval.py`. Args: `--checkpoints`, `--task {snli,spam}`, `--output_dir`, `--freeze_base`, `--max_train_samples`, `--results_json`.

The model wraps the causal LM with a linear head pooled over the last non-padded token's hidden state (appropriate for decoder-only architectures).

### Task 8 — Hard-Negative Strategy Ablation

If the contrastive approach continues to look promising after Tasks 6–7, this ablation identifies which WordNet strategy drives the result. Three ablation models, each using only one negative type:

1. **Co-hyponym only** (`--strategy co_hyponym`): siblings in the hierarchy (semantically similar but wrong subtree)
2. **Wrong-sense distractor only** (`--strategy wrong_sense`): different synsets of the same word (Chen's recommended safer negatives)
3. **Same-POS fallback only** (`--strategy same_pos`): other WordNet words with matching POS

`build_contrastive_dataset.py` now supports `--strategy {all,co_hyponym,wrong_sense,same_pos}`. Compare ablations against the full contrastive model (all three strategies) and both baselines.

## Planned Rigor for ACL/EMNLP Submission

Beyond Tasks 5–8:
- **Multiple seeds**: 3 seeds for the main comparison (CLM, Syn-NCP, Contrastive) — report mean ± std
- **Scale**: Pythia-1.4B after Llama-1B results are validated (Chen's recommendation: last)
- **Hyperparameter grid**: α ∈ {0.25, 0.5, 1.0}, β ∈ {0.5, 1.0, 2.0} for the contrastive model
- **Coverage stat**: always report WordNet coverage (fraction of concept slots with ≥1 hard negative) alongside any contrastive result
- **Qualitative error analysis**: examples where the contrastive model re-ranks concepts correctly that NCP misses

## Models

- **Llama-3.2-1B** (`meta-llama/Llama-3.2-1B`) — primary (T4 GPU compatible, ~5GB VRAM). BOS=128000, EOS=128001.
- **Pythia-1.4B** (`EleutherAI/pythia-1.4b`) — secondary for scaling analysis (Task 9, deferred until Llama-1B results validated). Pythia uses different BOS/EOS token IDs than Llama — verify before running concept eval scripts. Load with `AutoModelForCausalLM`.
- 8B models are OOM on T4; usable on A100 (Colab Pro).

## Common Pitfalls

- `context_syn` column must be parsed with `ast.literal_eval()` — it is stored as a stringified Python list
- The `completions_lookup` / `positives_lookup` / `negatives_lookup` dicts are built at tokenization time, not training time — building happens in the data pipeline of each `run_clm_*.py` script
- Pad token must be added: `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`
- BOS=128000, EOS=128001 are Llama-3 token IDs; Pythia uses different IDs
- WordNet coverage on informal YouTube text is expected to be <100% — always report the coverage stat
- Chen's note: co-hyponyms may still be valid completions in context — wrong-sense distractors are safer hard negatives (but June's ablation showed nothing about them: mining was broken; see REVISED ROUND)
- Use `data/*/youtube_clean/` splits and `eval_concept_ppl_v2.py` for anything new; the June splits/metric are deprecated
- "coverage" means two different things: negative-mining coverage (builder) vs eval-slot coverage (eval v2) — never conflate them in tables

## Compute Notes

- Use `--gradient_accumulation_steps 8 --per_device_train_batch_size 2` on T4
- `--bf16 True --torch_dtype bfloat16` required for memory efficiency
- `--auto_find_batch_size True` as safety net for OOM
- EarlyStoppingCallback with patience=3 is configured in all trainers

## Colab Notebook

`concept_aware_training_colab.ipynb` is the primary run environment. Key variables set early in the notebook and reused throughout:
- `MODEL_LOCAL_PATH` — local path to downloaded HF model
- `OUTPUT_ROOT` — Drive-backed output directory
- `DRIVE_DATASET_ROOT` — `"/content/concept_aware_training/data/"`
- `SYN_CUSTOM_TRAIN` / `SYN_CUSTOM_VAL` — synonym CSV paths
- `HYP_CUSTOM_TRAIN` / `HYP_CUSTOM_VAL` — hypernym CSV paths
- `CONTRASTIVE_TRAIN` / `CONTRASTIVE_VAL` — Task 3 output paths (set in Task 3 cells)

Cells 32-43 cover Tasks 1-4 (inserted before the Troubleshooting section at cell 44).

## Git Conventions

- Remote: `SharvaGogawale1/concept-aware-training` (Sharva's fork, shared with Madhura and Chen)
- Push with `gh` CLI or standard `git push origin main`
- Do not commit: model checkpoints, wandb/ directories, large output folders
