# Round 2 protocol

> Approved and committed before any test pass of a new arm; `scripts/round2/evaluate_round2.py`
> refuses a test pass unless this file is committed, unmodified and approved. Changing it later
> needs a new commit that says what changed and why.

## 0. Scope of this round (decided 2026-10-17)

Trained now (Tier 1, 30 runs): **Alt-Balance** at γ ∈ {0.0625, 0.125} × seeds {42, 123, 2024} × 3
models (18 runs), and **Alt-Mass** seeds 123 and 2024 at both γ × 3 models (12 runs). The October
Alt-Mass runs exist at seed 42 only, so comparing the four cells of the 2×2 fairly needs its other
two seeds as well; with them, every cell has three seeds. Concept loss and Alt-Mass+Balance reuse
the October checkpoints (three seeds each; Task 1 test B licenses the reuse of the balanced runs).

Not in this round unless the authors add them: the uncapped control, the member-wise (Iyer et al.)
baseline, the coefficient sweep, scoring on the sealed holdout, and the 8B scale-up. Sections below
that describe them stay as written so that adding them later does not change this protocol's rules.

- Protocol version: 1
- Approved by: the repository owner (git user SharvaGogawale1), 2026-10-17, in the working
  session, after Tier-1 training and before any scoring of the new runs. PI sign-off is not
  recorded here.
- Git hash: the commit that removes this file's TEMPLATE banner (`git log -1 -- ROUND2_PROTOCOL.md`).
- Trainer patch SHA-256 (`external/learning-concepts.patch`): bd7d72c7dfff2e3b570794ed47e78034416afe97607b2a531b51b18306285137
  (the patch every Tier-1 run was trained with; recorded per run in `round2_run.json`).

## 1. Objectives (fixed by the brief)

At a supervised position $t\in\mathcal I$: concept set $C_t$, observed token $y_t$,
$A_t=C_t\setminus\{y_t\}$ (deduplicated by the existing preprocessing), $M_t(X)=\sum_{x\in X}p_t(x)$
in log space, $q_t(a)=p_t(a)/M_t(A_t)$.

| term | definition | flag |
|---|---|---|
| ALT MASS | $R_t=-\log M_t(A_t)$ | `--gamma-mass` |
| ALT BALANCE | $B_t=\mathrm{KL}(\mathrm{uniform}(A_t)\Vert q_t)$ | `--gamma-balance` |
| UNCAPPED CONCEPT | $S_t=-\log M_t(C_t)$, every $t\in\mathcal I$, no $\tau$ cutoff | `--gamma-uncapped` |

Loss: $L_Z+\frac1{|\mathcal I|}\sum_t(\gamma_mR_t+\gamma_bB_t+\gamma_uS_t)$, with $L_Z$ unchanged
(Zhang et al.: $\lambda=1$, $\tau=0.6$, normalized by $|\mathcal I|$). $|A_t|=0$: $R_t=B_t=0$;
$|A_t|=1$: $B_t=0$.

| objective | display name | $\gamma_m$ | $\gamma_b$ | $\gamma_u$ |
|---|---|---|---|---|
| `concept` | Concept loss | 0 | 0 | 0 |
| `alt_mass` | Alt-Mass | γ | 0 | 0 |
| `alt_balance` | Alt-Balance | 0 | γ | 0 |
| `alt_mass_balance` | Alt-Mass+Balance | γ | γ | 0 |
| `concept_uncapped` | Concept loss + uncapped | 0 | 0 | γ |

`alt_mass_balance` reuses the October balanced (`--alt-aux uniform`) checkpoints: Task 1
test B shows $\nabla U=\nabla(R+B)$ through the training-loss code path.

## 2. Weights, models, seeds (fixed by the brief)

- Primary weight γ = 0.125; sensitivity γ = 0.0625. Coefficient-sensitivity grid (dev only)
  γ ∈ {0.03125, 0.25, 0.5}, first seed.
- Models: Llama-3.2-1B, Qwen3-1.7B-Base, Falcon3-1B-Base.
- Seeds: 42, 123, 2024 (recovered in Task 0; identical to the October runs).
- Shared settings: 4-bit NF4 QLoRA, r = 4, lora_alpha = 4 (PEFT scaling alpha/r = 1),
  LR 7e-5, 8 × 2 = 16 effective batch, 5 epochs (Augmented NTP: 1), final checkpoint.

## 3. Primary endpoint

Mass main effect and Balance main effect on **sealed-holdout GAP** (left scorer), at γ = 0.125,
per model:

- Mass main effect $=((Y_{10}-Y_{00})+(Y_{11}-Y_{01}))/2$
- Balance main effect $=((Y_{01}-Y_{00})+(Y_{11}-Y_{10}))/2$

where $Y_{mb}$ is the cell with mass on/off ($m$) and balance on/off ($b$). Inference:
paired hierarchical bootstrap, 10,000 replicates (resample matched seeds with replacement,
then targets with replacement, method pairing preserved). Holm–Bonferroni across the 3 models,
separately for each of the two main effects. α (family-wise) = 0.05.

**This round** does not score the sealed holdout (§0). As the brief specifies for that case, the
report opens with the same block computed on SWORDS test and marks it *secondary*: the October
test sets were read before the new arms existed (ledger `added_after_first_test_read=true`).

## 4. Key secondary endpoint

Alt-Mass − (Concept loss + uncapped), γ = 0.125, on sealed-holdout GAP, $M(A)$ and observed
share; same bootstrap; Holm across the 3 models.

## 5. Other secondary endpoints and comparison families (Task 6)

Families (Holm within each family, across model × dataset cells; never pooled across families):

1. each arm vs Concept loss;
2. Mass+Balance vs Mass; Mass+Balance vs Balance; Mass vs Balance;
3. Alt-Mass vs Concept loss + uncapped;
4. Member-wise vs each 2×2 cell;
5. factorial effects (main effects, interaction) per model, dataset, metric, γ.

Metrics: GAP, accepted-alternative NLL, rejected share, STS, C4 NLL, open-vocabulary Hit@50,
$M(A)$, KL$(u\Vert q)$, observed share. Datasets: SWORDS test, CoInCo-800 test, SemEval-07
(left and full scorers), held-out C4, STS, bm-semlex, six-task harness. Old test-set passes of
new arms are flagged `added_after_first_test_read=true` and are secondary by construction.

## 6. Member-wise (Iyer et al.) baseline (Task 4)

**Not in this round (§0).** If added later: mapping in `docs/iyer_mapping.md`, approved after
reading; α grid with the budget the balanced γ received before October, one seed, per model
(Task 0.3: Llama 4, Qwen 4, Falcon 2).
- Selection rule (fixed by the brief): the α with the highest SWORDS-dev GAP subject to
  mean STS ≥ Concept-loss mean STS − δ.
- δ = 0.005 (the October gate's STS allowance), if the baseline is added.
- If Iyer et al.'s weighting is fully specified, it is used as is and no α is tuned.

## 7. Decision table and inclusion rules

**Decision table: none pre-declared for this round.** The research plan's decision table
(§§4–5 of `01_research_plan_v3.md`) was not available when this was written. This round
therefore *estimates* the factorial effects and reports them with their intervals; it
selects no arm and declares no winner from them. A decision table must be added here, by a
new commit, before any sealed-holdout pass.

**Inclusion rules (apply to every table of this round):**
- Runs: every Tier-1 run with ledger status `ok`, and the October checkpoints listed as
  `reuse` by `run_round2.py plan --tier 1`. A failed or diverged run is retried once with the
  same seed; if it fails again it is reported as failed, never replaced by another seed.
- Seeds: 42, 123, 2024 in every cell; effects are seed-matched.
- Targets: all targets each evaluator scores, identical for every arm of a model (Task 0
  verified this for the October files); each table states its eligible count, including the
  reduced counts of accepted-alternative NLL and rejected share (prefix collisions).
- Scorers: the left-context scorer is the main one; the full-context scorer is reported beside it.

## 8. Sealed holdout

2,000 CoInCo targets, `sealed/sealed_holdout_ids.json` (committed before this protocol).
Evaluated once, for every arm listed in `sealed/FREEZE.md`, in one pass; the evaluator refuses
without `FREEZE.md` and writes a lock afterwards.
