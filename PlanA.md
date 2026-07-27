# Plan A — "Concept signal belongs in the data, not the loss"

**Status:** proposed 2026-07-20. Target: submission-ready draft in ~4 weeks (ICLR 2027 is the
realistic venue — EMNLP 2026 ARR has passed; verify exact ICLR abstract/full deadlines, historically
mid/late September). All experiments on `data/*/youtube_clean/` with `eval_concept_ppl_v2.py`.

---

## 1. Thesis and claims

**Framing correction (2026-07-20, after Chen's reply + a full code audit).** Chen was right to
distrust the July "negative" result. A deeper audit found **three** training-side bugs (§1.1), two of
them new. Their combined effect is that **the July master table does not report a fair test of the
loss-based objective** — so the honest current status is *not* "the loss approach fails." It is:
**the loss-based objective has never been evaluated without bugs.** The whole point of this plan is to
run that clean test (arm **A2**) and only then decide what the paper claims. The eval itself
(`eval_concept_ppl_v2.py`) was audited and is **correct** — it is model-agnostic and simply detects the
damage the training bugs cause, so it is not the source of the anomaly Chen flagged.

The paper is organized around three claims. Each has a named test, a pre-registered success criterion,
and a fallback interpretation, so the work is publishable under every outcome — but the C2 wording is
now *conditional on A2*, not asserted from the buggy runs.

| # | Claim | Current (post-audit) status | What's needed |
|---|-------|------------------|----------------|
| **C1** | Concept signal injected **via data augmentation** improves concept coverage (set mass / concept PPL) at little or no NTP cost | `standard_clm` (actually the data-aug arm) leads every intrinsic metric, but there is **no pure-vanilla control**, so "augmentation helps" is unproven | Arm **A0** (CLM on `vanilla_train.txt`) ×3 seeds |
| **C2** | Whether concept signal injected **via the loss** can work is **currently untested**, because every loss run to date is bug-contaminated (§1.1). The paper's contribution here is to (a) *diagnose* why the naive implementations failed, and (b) *test a corrected implementation* | July losses are real but **confounded**: 100–500× worse concept PPL / 2–3× worse NTP — produced by models with broken objectives and/or PAD contamination. Not usable as evidence that the objective itself fails | Diagnostics **D1/D2** to demonstrate the mechanisms + arm **A2** to test the corrected objective |
| **C3** | The decisive test: a **bug-free** set-marginal loss (leading-space first-token supervision, PAD-masked labels, forgetting mitigations) compared against augmentation head-to-head | Trainers now fixed (§1.1, §5.1); no clean run exists yet | Arm **A2** (fixed set-marginal trainer), ×3 seeds |

**Branch logic (decided at the end of Week 2, from A2 vs A1 with paired stats):**
- **A2 ≥ A1:** positive method paper — "set-marginal concept supervision, implemented correctly,
  matches or beats data augmentation; here is why three prior implementations (incl. the source
  paper's) silently failed." Strongest outcome.
- **A2 < A1 but A2 ≫ the buggy loss runs:** the honest, still-interesting result — "concept signal is
  most reliably delivered through data; even a corrected loss underperforms augmentation, and here is
  the mechanism." The corrected A2 (not the buggy runs) is the evidence. Analysis-track viable.
- **A2 ≈ buggy loss runs (fix changes nothing):** treat as a red flag to re-examine A2's
  implementation *before* concluding — a null after a real fix is suspicious at this scale.
- **A1 ≈ A0 (augmentation doesn't help either):** pivot to "what does concept post-training actually
  do to a 1B model?" carried by D1/D2/D3 + the representation battery (§7). Weakest; see Risks.

### 1.1 The three training bugs (all verified in code 2026-07-20)

| Bug | Where | What it does | Fixed? |
|-----|-------|--------------|--------|
| **#1 — Bare-token convention** | `contrastive_trainer.py`, `differentiable_ncp_trainer.py` (`_get_single_token_id`); also `custom_trainer.py:45`, `hierarchical_trainer.py` | Concept words tokenized with **no leading space** and multi-token words dropped. Under BPE `"cat"` ≠ `" cat"`, so the loss put mass on token ids that never occur mid-sentence — exactly what makes v1 eval circular and v2 eval (correctly) show a loss. | **Yes** — now uses first token of `" "+word`, deduped; multi-token supervised on first continuation token. Mirrors `eval_concept_ppl_v2`. |
| **#2 — PAD-label contamination** | ALL FOUR run scripts' `tokenize_function` (e.g. `run_clm_syn_custom_loss.py:565`) | `padding="max_length"`=128 + `attention_mask=[1]*len` (attends pads) + `labels=ids.copy()` (HF only ignores −100). Median context 12 words ⇒ **~88% of supervised positions train "predict [PAD]"**, and the concept slot's own CLM label is the first `[PAD]` — so CLM and NCP terms fight at the slot. Explains train loss 0.19–0.53 and "eval acc" ~0.90 (PAD accuracy). The **CLM baseline is unaffected** — `run_clm.py` uses `group_texts` (packed, no padding) — which is exactly why CLM out-scores the concept models (answers Chen's "another bug"). | **Yes** — pads now masked: `attention_mask=0` and `labels=−100` at pad positions; `input_ids` unchanged so lookup keys still match. |
| **#3 — Zero-gradient concept term** | `custom_trainer.py:57` (`torch.no_grad()`, subtracts a constant) and `hierarchical_trainer.py:82,105,121` | The concept adjustment is computed under `no_grad` and added/subtracted as a **detached constant** — zero gradient to the weights. Also scores `logits[0,-2]` (batch elem 0 only, a pad/EOS position). **Consequence: the revised-round `syn_ncp` and `hyp_ncp` are NOT concept-trained — they are just CLM fine-tuned on the concept CSV (with bug #2).** | **Not "fixed" — retired.** The correct positives-only NCP objective already exists as `differentiable_ncp_trainer.py`; use **diff_ncp as the canonical corrected-NCP arm** and **drop syn_ncp/hyp_ncp from headline tables** (relabel as "CLM on concept CSV"). |

**Implication for interpretation:** of the seven revised-round models, only `diff_ncp` and the
`contrastive_*` variants ran a gradient-flowing concept objective at all — and those still carried bugs
#1 and #2. `syn_ncp`/`hyp_ncp` never had a working objective. This is why the July table cannot
support a claim about the objective in either direction, and why A2 is a prerequisite, not a
nice-to-have.

---

## 2. Theory check — does the theory support each claim?

### C1: why augmentation should help (and one honest caveat)
Augmenting each sentence with n synonym-substituted variants trains standard CE toward the
*empirical distribution* over set members at the slot — probability mass is spread across the set in
the **natural leading-space token convention**, and the model also gets full-sentence signal
(tokens after the slot). This is exactly the set-marginal effect the loss variants were chasing, but
delivered on-distribution. The source paper's own appendix supports this: every intrinsic win in
Table 5 is a Data-Aug variant.

**Caveat (must be stated in the paper):** `vanilla_train.txt` is count-matched by *upsampling*
(7,859 lines, only 545 unique sentences, ~14× repetition — the paper's "No Concept" design), while
the augmented file has 7,257 unique lines. So part of any A1-over-A0 gap may be diversity /
anti-overfitting rather than concept signal per se. This is inherited from the original paper's
design; we report it and (optionally, arm **A1h**) show the effect replicates on the hyp-augmented
text. Do not claim more than the design supports: A1 vs A0 tests "concept-augmented variants vs
repeated originals at matched token count."

### C2: why the loss variants had to fail
- **Convention mismatch:** under Llama BPE, `"tactical"` and `" tactical"` are different token ids.
  The trainers (`_get_single_token_id`) supervise the bare id at the position after the context —
  an id that essentially never occurs in running text at that position. Optimizing it (i) moves mass
  onto off-distribution tokens (concept PPL under the natural convention *must* get worse), (ii)
  competes directly with the CLM term at the same position (NTP degrades), and (iii) silently drops
  every multi-token concept, so effective supervision coverage is far below the reported mining
  coverage.
- **PAD-label contamination (found 2026-07-20, prompted by Chen's "another bug" hunch):** all four
  concept run scripts pad every row to 128 tokens, attend over the PADs (`attention_mask=[1]*len`),
  and set `labels = ids.copy()` — HuggingFace CLM loss only ignores label −100, so the model is
  explicitly trained to predict `[PAD]` at every pad position. Median context is 12 words → **~88%
  of the CLM training signal per row is "emit [PAD]"**. Worse, the concept slot (last real token) has
  `[PAD]` as its own CLM label, so at the exact position where the NCP term pushes toward bare-token
  synonym ids, the CLM term pushes toward `[PAD]` — and *neither* is a natural continuation. This
  predicts a PAD attractor in the output distribution after any prompt, depressing all natural-text
  probabilities — measured as both worse NTP PPL and worse concept PPL. Symptoms already in the logs:
  train losses of 0.19–0.53 (mostly easy PAD prediction) and "eval accuracy" ~0.90 (PAD accuracy).
  The June-v1 "gains" and July-v2 "losses" are the same facts measured in two conventions — models
  reached ~0.90 accuracy on their own convention while NTP acc fell to ~0.42.
- **Catastrophic forgetting:** full fine-tuning of a 1B model on ~1.8–2.4K short prefix rows,
  3 epochs, LR 5e-5 (default; visible in run logs), no replay or regularization. Predicts the
  observed NTP PPL degradation (16.6 → 30–48) partly independent of the concept term — D2 separates
  the contributions.
- **Why the eval is NOT the bug (answer to Chen's question):** four independent lines of evidence.
  (1) NTP PPL — a completely standard metric on plain text, separate code path from v2 — degrades
  2–3× for every concept model; (2) the source paper's own Table 5 shows the identical pattern at
  8B; (3) both seeds of the same model preserve the CLM-beats-NCP ordering; (4) the two training
  bugs above are verified in code and *mechanically require* this outcome. D1 turns this into a
  figure.
- **Seed variance:** one seed pair already showed 3× concept-PPL and 18-point NTP-PPL swings
  (contrastive_merged seed 42 vs 123). Theory: tiny dataset + full FT = high-variance endpoint. All
  cross-model claims need ≥3 seeds and *paired* per-slot statistics (unpaired CIs span two orders of
  magnitude and can resolve nothing).

### C3: why the fixed loss *could* beat augmentation (the interesting hypothesis)
The set-marginal objective `-log Σ_c p(" c" | context)` maximizes **total** set mass but is agnostic
about the distribution *within* the set — it permits the model to keep a favored surface form.
Augmentation instead forces mass-spreading toward uniform over the set (which necessarily raises the
NLL of any single gold token). So theory predicts a fixed set-marginal loss can achieve **equal or
better set mass with less NTP damage** — a strictly weaker constraint on the model. That is a real,
falsifiable advantage and the paper's central question. The honest competing prediction: augmentation
also supervises post-slot continuations and needs zero custom machinery, so it may win on general LM
quality regardless. Either result is informative — this is what makes the experiment decisive rather
than incremental.

---

## 3. Experimental arms

Naming: **A\*** = training arms, **D\*** = diagnostics (cheap/no training). Seed policy in §5.

| Arm | What | Data | Objective | Status | Runs needed |
|-----|------|------|-----------|--------|-------------|
| **A0** | Pure-vanilla CLM control | `syn/youtube_clean/vanilla_train.txt` (upsampled originals) | standard CLM (`run_clm.py`) | **missing — highest priority** | seed 42 now; ×3 at camera-ready |
| **A1** | Data augmentation (= existing `standard_clm`, relabel it in all tables) | `syn/youtube_clean/context_syn_train.txt` | standard CLM | seed 42 exists ✅ | +2 seeds deferred |
| **A2** | **Fixed** set-marginal NCP (the decisive arm) | clean syn concept CSV + vanilla replay rows | CLM + α·(−log Σ p(" c"\|ctx)), leading-space first-token ids (deduped), LR 1e-5, 2 epochs | **trainer patch applied 2026-07-20** (bugs #1+#2, §5.1); replay + LR still to wire | seed 42 now; ×3 at camera-ready |
| A3 | Fixed contrastive (β>0 on top of A2) | A2 data + negatives | A2 + InfoNCE | optional — only if A2 shows life AND budget remains (July ablations showed negatives ≈ no-negatives) | 0–1 |
| A1h | Augmentation replication on hyp side | `hyp/youtube_clean/context_syn_train.txt` | standard CLM | optional robustness check for the upsampling caveat | 0–1 |
| **D1** | **Convention + PAD mass probe (smoking-gun figure)** — at each val slot, per model, compare: p([PAD]), Σ p(bare-token ids), Σ p(leading-space first-token ids) for the concept set | syn+hyp val CSVs | eval-only script | missing | 0 GPU-hours (~min) |
| **D2** | Forgetting decomposition: same trainer/pipeline as the broken runs but α=0, β=0 (pure CLM on the concept CSV rows) | clean syn concept CSV | CLM only | missing | 1 run |
| **D3** | **Manual audit of positives** — 50 syn-val + 50 hyp-val slots: is each "positive" a plausible in-context continuation? | val CSVs | human, ~1 hr | missing — **gates everything** (see Risks) | 0 GPU |
| — | Existing broken-convention arms (syn_ncp, hyp_ncp, diff_ncp, contrastive_\*, ablations) | — | — | keep as-is; they ARE the C2 evidence. **Train no more of them.** | 0 |

### Predicted result patterns (pre-registered)

| Comparison | Predicted (theory) | If instead… | Interpretation |
|---|---|---|---|
| A1 vs A0, paired per-slot concept NLL | A1 better, CI excludes 0; NTP PPL similar or A1 better | A1 ≈ A0 | augmentation effect was diversity/training generally, not concept signal → fall back to diagnosis framing |
| A1/A0 vs base (untrained) Llama-1B on concept NLL | both trained arms better than base | base ≈ A1 | post-training adds nothing at this scale → strong (uncomfortable) analysis result; add base model row to all evals — it's free |
| D1: bare-vs-spaced mass | loss-trained models put ≫ mass on bare ids; A0/A1/base ≈ 0 | no difference | convention hypothesis wrong → investigate before writing anything |
| D2 (α=0 on CSV rows) NTP PPL | between A1 (16.6) and syn_ncp (35.9) — quantifies forgetting share vs objective share | ≈ 35.9 | degradation is data-format/forgetting, not the NCP term → adjust C2's wording |
| A2 vs syn_ncp/diff_ncp | A2 dramatically better concept PPL (convention fixed) and better NTP (LR/replay) | A2 ≈ diff_ncp | the fix didn't take — check implementation before concluding anything |
| **A2 vs A1 (headline)** | open — either direction publishable | — | decides positive vs negative framing (§1) |
| A2 set mass vs A1 set mass, at matched NTP PPL | the theory-favored win condition for A2 | — | report as the trade-off frontier plot (set mass vs NTP PPL, one point per arm per seed) |

### Downstream battery (Task 7 continuation — currently the only positive signal)
- **Keep:** SNLI linear probe + low-resource FT. July result to defend: at n=500 *every* concept
  model (0.70–0.74) beat CLM (0.645). Single seed → must be seeded before it can carry weight.
- **Runs:** probe + FT@n=500 for {A0, A1, A2} × 3 seeds (9 probes + 9 FTs). Full n∈{100, 500, 1000}
  curve for seed 42 only. SPAM: drop from the paper body (saturated; footnote at most).
- **Caution:** the July low-res winners were the *broken-convention* models. If the effect is real it
  may come from representation perturbation, not concept knowledge — the A0/A1/A2 versions of this
  table will say. Do not headline the July numbers.

### Representation & capability-retention battery (added 2026-07-20, per Chen's feedback)
All eval-only — no training cost. Run on {base, A0, A1, A2, syn_ncp, contrastive_merged} so the
broken arms serve as the "damaged" reference points.

- **R1 — Synonym-invariance gap (the targeted representation metric; highest value).** Using our own
  clean val data: embed (mean-pool final hidden states) each original sentence, its
  synonym-swapped variant (from the augmentation files), and a hard-negative-swapped variant (from
  the contrastive CSVs). Report `gap = mean cos(orig, syn-swap) − mean cos(orig, neg-swap)`, paired
  per sentence across models. This tests the paper's central claim at the *representation* level,
  independent of the (damaged) output head. **Unifying hypothesis it can confirm:** concept
  objectives reshape representations (→ the low-resource SNLI edge, probes) while corrupting the
  output distribution (→ the PPL losses). If syn_ncp wins R1 while losing PPL, that narrative holds;
  if A2 wins R1 *without* the PPL damage, the positive framing is fully assembled.
- **R2 — Standard STS (external validity for R1).** STS-B (+ SICK-R if cheap) via cosine of pooled
  hidden states, Spearman correlation. Absolute numbers will be modest (decoder-only 1B, no
  embedding training) — that is fine; only the *relative* ordering across checkpoints matters, same
  pooling protocol everywhere. Full MTEB is out of scope (overkill; absolute scores would
  misleadingly embarrass the models).
- **R3 — Capability retention (Chen's ARC/MMLU request, reframed).** lm-eval-harness on ARC-easy,
  ARC-challenge, HellaSwag, PIQA, Winogrande (+ MMLU for completeness). At 1B with 2.2K training
  rows, *gains* are not a realistic hypothesis — post-training on YouTube fragments adds no
  knowledge; the causal channel is damage. So these are **retention** metrics: prediction is
  A0/A1/A2 ≈ base, broken loss arms measurably below base. Note MMLU is near floor (~32%) for
  Llama-3.2-1B and noisy; the commonsense suite is the more sensitive damage detector at this
  scale. Retention joins set-mass and NTP PPL on the trade-off frontier plot.

---

## 4. Statistical protocol (non-negotiable — this is what makes the paper survive review)

1. **Paired per-slot analysis is primary.** `eval_concept_ppl_v2.py` already writes `per_row_nll`;
   row order is deterministic given the CSV + canonical tokenizer, so rows align across checkpoints.
   Write one analysis script (no GPU): for each model pair, per-slot ΔNLL → paired bootstrap 95% CI,
   win rate, and median NLL. Never lean on unpaired `exp(mean NLL)` CIs again — July's spanned two
   orders of magnitude.
2. **exp(mean NLL) stays in tables for continuity, but** report median-based concept PPL and set
   mass alongside; heavy-tail means are dominated by a few impossible slots.
3. **Seeds:** 3 × {A0, A1, A2} for every headline number; mean ± std. Single-seed numbers may
   appear only in appendix, marked.
4. **Coverage discipline:** negative-mining coverage (builder) and eval-slot coverage (v2) reported
   separately, always. Additionally report **training supervision coverage** for loss arms (fraction
   of positives that actually produced a usable token id) — the old single-token filter made this
   silently lower than mining coverage; A2 should report it too.
5. **Every intrinsic table gets a base-model (untrained Llama-3.2-1B) row.** Free, and anchors all
   claims about what post-training adds.
6. v1 and v2 concept numbers never appear in the same table. June numbers appear only in the
   audit/appendix section, clearly marked superseded.

---

## 5. Engineering tasks

### 5.1 Trainer fix for A2

**DONE 2026-07-20 (committed to the run scripts + trainers; rerun on Colab to take effect):**
- **Bug #2 — PAD contamination fixed in all four run scripts.** `tokenize_function` now returns
  `attention_mask=0` and `labels=−100` at every pad position; `input_ids` unchanged so the
  concept-lookup key `str([BOS]+padded+[EOS])` still matches in the trainers. (`group_texts` is dead
  code under non-streaming and was left as-is; note it if streaming is ever enabled.)
- **Bug #1 — convention fixed in `differentiable_ncp_trainer.py` and `contrastive_trainer.py`.**
  `_get_single_token_id` → `_get_concept_first_token_id`: first token of `" "+word` (leading-space,
  mirrors `eval_concept_ppl_v2._continuation_ids`), multi-token words supervised on their first
  continuation token (documented approximation; keeps zero-extra-forward-pass), and ids **deduped**
  before `logsumexp` (shared first tokens would double-count mass).
- **Bug #3 — NOT fixed, retired.** `custom_trainer.py`/`hierarchical_trainer.py` (the zero-gradient
  `no_grad` trainers) are not resurrected; `differentiable_ncp_trainer.py` is the correct
  positives-only NCP. Drop `syn_ncp`/`hyp_ncp` from headline tables (see §1.1).

**Still to wire before running A2:**
- **Forgetting mitigations:** `--learning_rate 1e-5`, 2 epochs, and **replay** — append vanilla
  sentences as rows with an empty concept set (CLM loss applies, NCP term skips them) at ~1:1 with
  concept rows. Keep α=0.5 initially; α ∈ {0.25, 0.5, 1.0} only if budget remains.
- **Startup log:** print training supervision coverage (how many positives yielded a usable first-
  token id) — this is a distinct stat from mining/eval coverage and must be reported.
- **Unit check (Week 1):** on 20 sampled slots, assert the trainer's supervised first-token id equals
  `eval_concept_ppl_v2`'s first continuation token — closes the loop that training and eval now share
  one convention.

**A2-full (stretch, only if week 3 has slack):** slot-aligned set-marginal loss on *full sentences*
(loss = CLM on vanilla sentence + α·set-marginal at the slot position). This is the theoretically
fair factorial counterpart to A1 (same sentences, signal via loss instead of data). ~1–2 days of
data-pipeline work; A2-lite is the budgeted version.

### 5.2 D1 probe script (~50 lines, eval-only)
For each val slot and each checkpoint: softmax at the slot position; sum mass over (a) bare-word
first-token ids, (b) spaced-word first-token ids, for the positive set. Output per-model means +
a two-bar figure. This is the paper's mechanism figure.

### 5.3 Paired-stats script (§4.1) — runs on existing JSONs today.

### 5.4 Hygiene
- Fix notebook cell 47 (prose mixed into a code cell — currently not runnable).
- Relabel `standard_clm` → "CLM + data augmentation (NSP-DataAug)" in every table; the current label
  is actively misleading and a reviewer who catches it will assume the worst.
- Drop `syn_ncp`/`hyp_ncp` from headline tables (bug #3: zero-gradient objective — they are CLM on the
  concept CSV, not NCP). Keep `diff_ncp` as the canonical corrected-NCP arm.
- New runs: `--save_total_limit 1`, strip optimizer state (Drive quota lesson already learned).
- Commit the trainer fix + scripts to the shared repo before any Colab run (Colab pulls from GitHub).

---

## 6. Budget & timeline (4 weeks from 2026-07-20)

Training runs: A0×3, A1×2, A2×3, D2×1 (+optional A3/A1h) ≈ **9–11 runs × ~20–40 min** on L4 —
about one Colab session. Downstream: 9 probes (~25 min each) + 9 low-res FTs (~15 min) + seed-42
curve ≈ 2 sessions. Diagnostics and stats: CPU/eval-only. Comfortably inside Colab Pro budget.

**Week 1 — foundations & gate**
- D3 manual audit (do FIRST — 1 hour, gates the metric's validity).
- Paired-stats script on existing JSONs (sharpens the C2 numbers immediately, no GPU).
- D1 probe script + run → mechanism figure v0.
- **Seed 42 only for now** (A0, A2, D2 — 3 runs; A1 seed 42 = existing standard_clm). Extra seeds are
  deferred (notebook cell 9c, `RUN_EXTRA_SEEDS=False`) — the seed-42 pass is enough for the Week-2
  A2-vs-A1 decision; error bars come at camera-ready. Base-model row added to evals.
- Trainer fix (bugs #1+#2) is **already applied**; Week-1 task is the unit check (supervised ids ==
  v2's first continuation tokens on 20 sampled slots) + wiring replay/LR for A2.

**Week 2 — the decisive arm**
- A2 seed 42 + D2 (extra A2 seeds deferred; decision is made on seed 42, de-risked later).
- Full intrinsic eval battery (v2 syn+hyp, paired stats, D1 on A2).
- **Decision point (end of week 2): positive vs negative framing.** Inform Chen with the A2-vs-A1
  paired result and the D1 figure.

**Week 3 — downstream + representation battery + slack**
- SNLI probe + FT@500 for {A0, A1, A2}×3 seeds; seed-42 full curve.
- R1 (synonym-invariance gap) + R2 (STS-B) + R3 (lm-eval retention suite) — all eval-only, ~1
  session total.
- Slack absorbs: A3 / A1h / A2-full / α sweep — strictly in that priority order, only if weeks 1–2
  held schedule.

**Week 4 — writing**
- Figures: D1 mechanism bars; set-mass-vs-NTP-PPL frontier; low-res curve; master table (3-seed
  mean±std, paired CIs).
- Sections: audit narrative (leak, circular metric, broken ablation) as the methodology
  contribution; June numbers appendix-only.
- Chen + Sharva review pass; framing conversation (see Risks).

---

## 7. Risks & mitigations

1. **D3 audit shows the positives are junk** (plausible: CLM set mass is only 3.9%, concept PPL 4.2K
   even for the best model — the gold sets are largely improbable continuations). Mitigation: build a
   filtered eval subset (slots where ≥1 auditor-approved positive); report both. If <~60% of slots
   have any valid positive, the concept-PPL metric itself becomes a paper finding ("the concept
   annotations don't support the task") and the LLM-generated-annotation quality becomes a section.
2. **A1 ≈ A0** — augmentation doesn't help either. The paper becomes pure analysis/diagnosis
   (C2 + D1/D2/D3 + audit). Still submittable, but tell Chen immediately; the upsampling confound
   (§2) cuts both ways and must be discussed.
3. **Seed instability persists at 3 seeds** (July showed 3× swings). Paired per-slot stats mitigate;
   if arms are still indistinguishable, report that honestly — "effects smaller than seed noise at
   this scale" is itself the finding, and motivates the deferred Pythia-1.4B scale point.
4. **Political:** the paper reframes the loss-function half of Chen's own paper as non-functional.
   Chen is both PI here and author there — this must be a *joint* framing decision, positioned as
   "explaining and extending Iyer et al.": their appendix already contains the same pattern; we
   supply the mechanism. Have this conversation at the week-2 decision point, not at submission.
5. **Timeline:** ICLR gives ~2 months, so a week of slip is survivable; EMNLP is gone. If A2-full
   becomes necessary for the story and doesn't fit, it goes to the appendix as "preliminary" or to
   camera-ready.

## 8. Explicitly out of scope (do not spend budget here)
- Any further training with the broken-convention trainers (ablations, hyp variants, α/β grids).
- Pythia-1.4B (Chen's ordering: last, and only after the Llama story is fixed).
- Cross-domain (YouTube→news/arXiv) — eval-only follow-up, camera-ready material.
- SPAM beyond the existing sanity numbers.
- Full MTEB (R2 covers the STS slice; full suite adds cost without changing conclusions).
- **MMLU — deferred (do NOT run now).** At Llama-3.2-1B, MMLU sits near floor (~32%, random 25%);
  arm differences will be within noise, so it cannot discriminate objectives and just burns ~1–2
  GPU-hr/model. Run the cheap, more-sensitive retention suite (ARC/HellaSwag/PIQA/Winogrande) now;
  add MMLU only at camera-ready for table completeness, and ideally at the Pythia-1.4B / larger scale
  where MMLU is meaningfully above floor. Sequencing: land the intrinsic (A2 vs A1) + representation
  (R1/R2) story first, then MMLU.

---

## 9. Chen's reply (2026-07-20) — analysis and response

**Meta-point that reframes the plan.** Chen is in skeptic mode, not agreement mode. "I suspect there
might be another bug… it doesn't make sense that CLM beats Syn-NCP" means *she does not believe the
negative result and thinks it's an artifact.* She is right. So the reply must NOT be "you're correct,
the loss approach fails" — it must be "you're correct, there was another bug (in fact two more); the
negative result is not trustworthy; here is the fix and the clean re-test." This is the §1 framing
correction, and it is the honest and defensible position.

**Point 1 — "another bug; CLM shouldn't beat Syn-NCP."** Correct, and the audit found **two** more
bugs, both training-side, not in the eval (§1.1):
- **PAD-label contamination** (bug #2): every concept run script trains ~88% of positions to predict
  `[PAD]` and puts `[PAD]` as the concept slot's own CLM label. The CLM baseline uses a different,
  clean code path (`run_clm.py` `group_texts`, no padding) — **that asymmetry is precisely why CLM
  out-scores Syn-NCP**, answering her exact question.
- **Zero-gradient objective** (bug #3): `custom_trainer.py`/`hierarchical_trainer.py` compute the
  concept term under `torch.no_grad()` and subtract it as a constant → `syn_ncp`/`hyp_ncp` were
  **never actually concept-trained**; they're CLM on the concept CSV. So the row she's comparing
  ("Syn-NCP") isn't what its name says.
- The v2 eval is **sound** (audited line-by-line: correct leading-space continuation scoring, correct
  set-marginal logsumexp, model-agnostic). Four independent checks corroborate: standard NTP PPL
  degrades in parallel, the source paper's Table 5 shows the same at 8B, the ordering is seed-stable,
  and all three bugs are verified in code.
- **Response artifacts:** (i) the D1 figure — per-model mass at val slots on `[PAD]` / bare-token /
  leading-space ids; (ii) 2–3 raw top-10 next-token lists (CLM vs syn_ncp after the same context) —
  most legible in email; (iii) the one-line code citations for each bug.

**Point 2 — "why would hard negatives increase this metric?"** They don't, measurably. The ablation
spread (1.19M–1.40M) is *smaller than the same model's seed swing* (0.40M vs 1.23M); all CIs overlap.
Correct claim: "no detectable effect." And with bug #3 in mind, the hard-negative ablations were run
on the contrastive trainer (which *does* flow gradients), so they're not zero-signal like syn_ncp —
but the effect is still below noise. Paired per-slot stats (§4.1) will state this properly. Do not let
the June "wrong-sense is best" narrative return. She noted this "relates to your second question" —
tie the answer back to whatever that thread question was when replying.

**Point 3 — "harder benchmarks (ARC, MMLU)? representation / STS from MTEB?"** Adopt both, reframed
(full spec in §7 R1–R3):
- **ARC/MMLU → capability RETENTION (R3).** Set expectations explicitly in the reply: at 1B with 2.2K
  YouTube rows, *gains* on ARC/MMLU are not a realistic hypothesis — the causal channel is damage, so
  these measure how much each objective preserves base capability. Note MMLU is near floor for
  Llama-3.2-1B; ARC/HellaSwag/PIQA/Winogrande are the sensitive detectors. Framing this now prevents a
  flat MMLU result from reading as a failed promise.
- **STS → representation direction (R2 + R1).** Her STS/MTEB idea is the sharpest suggestion and lands
  on the plan's one open puzzle: the buggy concept models *lose* every probability-space metric but
  *win* low-resource SNLI. R1 (synonym-invariance gap on our own data) + R2 (STS-B) test whether
  concept training improves *representations* even while damaging the *output head*. If so, that's a
  unified story and squarely in Chen's own research area (tokens-vs-thoughts). Full MTEB is out of
  scope; R2 covers the STS slice.

**Suggested reply skeleton (4 short paragraphs):** (1) "You were right — found two more bugs, both in
training not eval; the negative result isn't trustworthy and I've fixed them" + the three one-line
code cites and the CLM-vs-custom asymmetry; (2) hard negatives = no detectable effect, with the
seed-noise comparison; (3) A2 is the clean re-test now unblocked — what it is and when results land
(end of Week 2); (4) yes to ARC/MMLU (as retention) and STS (as R1/R2 representation), with the 1B
expectation-setting. Offer a call.
