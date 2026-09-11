# ComNetX IEEE Access Working Area

This directory is the canonical workspace for the IEEE Access version of
ComNetX. It uses the official IEEE Access LaTeX template dated May 13, 2026.

## Provenance

- The submitted ICDM 2026 source is `../../article/article.tex` at baseline
  commit `5715b7f`.
- The ICDM source, PDF, and figures remain in `../../article/` as a frozen
  conference snapshot. Existing ICDM scripts still write to that directory.
- The untracked `../../arxiv/` export is a local artifact, not the canonical
  source for the journal manuscript.
- The canonical measurements used by the conference paper remain in
  `../../results/icdm-2026-1/` and should be referenced rather than copied.

## Contents

- `main.tex`: journal manuscript.
- `references.bib`: verified bibliography.
- `figures/`: vector figures regenerated from validated evidence.
- `generated/`: LaTeX rows and macros emitted by the validation script.
- `analysis/validate_results.py`: asserted record selection, consistency
  checks, and figure/table generation.
- `requirements.txt`: pinned dependency for the validation and figure generator.
- `analysis/validated_results.json`: machine-readable evidence used by the
  manuscript.
- `literature-and-style-review.md`: journal requirements and style survey.
- `reviewer-notes.md`: curated reviewer-response record.
- `reviewer-audit-and-revision-plan.md`: status audit of the ICDM reviews and
  prioritized IEEE Access revision plan.
- `submission-checklist.md`: remaining author confirmations and upload checks.

The bibliography preserves all 67 works cited in the ICDM manuscript under
canonical descriptive keys. The journal revision adds only sources needed for
dataset provenance, evaluation definitions, and direct positioning; it does not
replace the reviewed conference bibliography with a shortened reading list.

## Working Rules

- Keep new measurements in a separately versioned bundle such as
  `../../results/ieee-access-2026-1/`; do not overwrite the ICDM bundle.
- Keep generated LaTeX build artifacts outside the source directory, for
  example under `../../tmp/pdfs/ieee-access/`.
- Use `reviewer-notes.md` as the bounded revision backlog. It intentionally
  excludes conference-only ratings and requests that cannot currently be
  supported by a realistic measurement plan.

Regenerate evidence before compiling the manuscript:

```sh
python3 -m pip install -r journal/ieee-access/requirements.txt
python3 journal/ieee-access/analysis/validate_results.py
```
