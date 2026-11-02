# IEEE Access manuscript and reproducibility artifact

Version: **ieee-access-r2-20261102**, 2 November 2026.

This directory is the complete manuscript distribution cited in Section VII.
The ZIP files contain the exact sources for this version, including the revised
manuscript, generated tables, and figures. Use these packaged sources when
reproducing this version; other working files in the research repository may
belong to a different revision.

- [Manuscript PDF](comnetx-ieee-access.pdf) — the revised 19-page article.
- [Manuscript source](comnetx-ieee-access-source.zip) — LaTeX, bibliography,
  template resources, generated tables and figures.
- [Evidence and code](comnetx-ieee-access-evidence.zip) — the manuscript source,
  table/figure generators, immutable archived results, ComNetX core, public
  Leiden adapter, configurations, tests, and reproduction documentation.
- [SHA-256 checksums](SHA256SUMS) — verify downloads with `sha256sum -c SHA256SUMS`
  (Linux) or `shasum -a 256 -c SHA256SUMS` (macOS).

Each ZIP also contains `ARTIFACT-MANIFEST.json` with a SHA-256 digest for every
packaged file. Datasets and optional external backend implementations are not
redistributed. In particular, S²CAG and native DF-Leiden need separately obtained
components for new experiments; their archived results can be regenerated as
tables and figures without those components. See `docs/reproducibility.md` and
`docs/third-party-components.md` inside the evidence archive.

To regenerate tables, figures, and the PDF from the evidence archive, extract it
into a new directory, install `requirements/ieee-access-evidence.txt` in a Python
virtual environment, and run `make evidence audit pdf`. A TeX distribution with
pdfLaTeX/BibTeX is required; use `TEX_BIN=/path/to/tex/bin` and
`PYTHON=/path/to/venv/bin/python` when needed. The source archive can instead be
compiled directly from `journal/ieee-access/`: pdfLaTeX, BibTeX, then pdfLaTeX
twice. These operations do not run community-detection experiments.

This is a versioned research snapshot, not a claim of journal acceptance.
It preserves the historical measurement provenance limits stated in the article.
The new paired S²CAG campaign that was running when this revision was prepared
is not included, and no unfinished results have been substituted for archived
measurements.
