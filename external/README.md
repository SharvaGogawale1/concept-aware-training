# Pinned Zhang et al. reproduction

The notebooks clone `https://github.com/christine-zhang1/learning-concepts` at
commit `b1d414143d11c8ed988b4cccbb06626cc8272bbe` into
`/content/learning-concepts`, then apply `learning-concepts.patch`.

The patch keeps the released set-marginal objective as the default. Its opt-in
changes add explicit seeds and output paths, checkpoint-free runs, JSONL
learning curves, uniform and token-level contrastive objectives, explicit PEFT
adapter evaluation, conservative negative mining, concept-data audits, and the
directional hierarchy trainer/evaluators.

The local checkout at `external/learning-concepts/` is intentionally ignored.
To refresh the patch after an upstream-code edit, stage changes in that nested
checkout and export its cached diff to `external/learning-concepts.patch`.
