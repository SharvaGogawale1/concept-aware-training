# LaTeX source

ACL style. Build with:

    latexmk -pdf main.tex

`main.tex` is self-contained: all four tables are inlined, so there are no
`\input` fragments to keep in sync. Every number comes from the Task 14 run
recorded in `notebooks/research_tasks_14_external_ntp_vs_ncp.ipynb`
(commit 3eafec0), sections 7b and 7c.

Set `\usepackage[final]{acl}` for camera-ready, or `[preprint]` for a
non-anonymous version with page numbers. The current setting is `[review]`.

Files `acl.sty` and `acl_natbib.bst` are the official ACL template files.

## What is written

Methods and Results only, plus a Limitations section, which ACL requires and
which does not count against the page limit. Abstract, introduction, related
work and Figure 1 are not written yet.

## Numbers that still need to change before submission

- `iyer2026beyond` in `custom.bib` is a placeholder. Fill in the real author
  list and venue.
- The alpha value was chosen on the same development set the tables report, so
  the tuned rows are optimistic. A held-out SWORDS test measurement is still
  owed and is gated in the notebook behind `RUN_PHASE_C_SWORDS_TEST`.
- The gold-exclusive result rests on seed ranges, not a paired bootstrap. The
  comparison is cheap to add: both arms are already trained and evaluated.
