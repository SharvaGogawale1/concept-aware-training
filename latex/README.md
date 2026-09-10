# ACL paper draft

Build from this directory with:

    latexmk -pdf main.tex

`main.tex` uses the official ACL style files already in this directory. The
reported results come from `notebooks/research_tasks_14_external_ntp_vs_ncp.ipynb`
and its completed alpha sweep.

## Paper focus

The main table keeps five interpretable systems: the untouched model, data
augmentation, the printed concept objective, a differentiable reconstruction of
the code-form objective, and the selected retention-aware method. Set-marginal
and contrastive variants appear only in the appendix because they do not improve
the central comparison.

## Required before submission

- Run paired bootstrap intervals for the selected alpha-4 method against data
  augmentation and the code-form reconstruction.
- Lock alpha at 4, train on the complete SWORDS development set, and evaluate
  the official test set once.
- Add another model scale or family after the 1B test result is confirmed.
- Confirm the description of the source implementation with its authors. The
  present text is limited to the public commit cited in the paper.

Use `\usepackage[final]{acl}` only for camera-ready output. The current draft
uses review mode and anonymous authors.
