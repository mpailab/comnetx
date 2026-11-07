# IEEE Access manuscript and reproducibility artifact

This submission version reports the random-feature sensitivity analysis using seeds 2–5. Each of these seeds has one complete execution per graph.

This directory is the complete manuscript distribution cited in Section VII.
The ZIP files contain the exact sources for this version. Use these packaged
sources when reproducing the journal article; other working files in the
research repository may belong to an earlier revision.

- [Manuscript PDF](comnetx-ieee-access.pdf): the revised 16-page article.
- [Manuscript source](comnetx-ieee-access-source.zip): LaTeX, bibliography,
  template resources, generated tables, and figures.
- [Evidence and code](comnetx-ieee-access-evidence.zip): manuscript sources,
  table/figure generators, immutable archived results, ComNetX core, public
  Leiden adapter, configurations, tests, and reproduction instructions.
- [SHA-256 checksums](SHA256SUMS): verify downloads with `sha256sum -c SHA256SUMS`
  (Linux) or `shasum -a 256 -c SHA256SUMS` (macOS).

The completed S²CAG campaign contains 20 full/ComNetX pairs, 40 method
executions, 300 per-update records and label arrays, and two shared initial
hierarchies. The evidence includes the executed configuration, environment,
source hashes, and a portable provenance subset. Its manifest maps 302 logical
NPZ paths to 61 unique objects, preserving every original file's exact bytes.
The verifier reconstructs the paired summary without running any detector.

Each ZIP contains `ARTIFACT-MANIFEST.json` with a SHA-256 digest for every
packaged file. Datasets and optional external backend implementations are not
redistributed. S²CAG and native DF-Leiden need separately obtained components
for new experiments; archived results and the manuscript can be regenerated
without them. The paper and artifact explicitly retain the limitations of the
historical measurements. See `docs/reproducibility.md` and
`docs/third-party-components.md` inside the evidence archive.

To regenerate tables, figures, and the PDF, extract the evidence ZIP into a new
directory, install `requirements/ieee-access-evidence.txt` in a Python virtual
environment, and run `make evidence audit pdf`. A TeX distribution with
pdfLaTeX/BibTeX is required; use `TEX_BIN=/path/to/tex/bin` and
`PYTHON=/path/to/venv/bin/python` when needed. Alternatively, compile the source
ZIP directly from `journal/ieee-access/`: pdfLaTeX, BibTeX, then pdfLaTeX twice.
These operations do not run community-detection experiments.

This is a versioned research manuscript, not a claim of journal acceptance.
