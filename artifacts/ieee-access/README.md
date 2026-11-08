# IEEE Access manuscript and reproducibility artifact

The paper reports how much contraction reduces the graph supplied to Leiden. The S²CAG comparison now includes the quality of keeping the initial Leiden partition fixed on each final graph; the evidence archive provides the calculation and the saved scores in `results/ieee-access-2026-4/s2cag-initial-partition-control.json`. The original timed runs and their execution provenance are unchanged.

The server adapter in this research checkout also handles a one-vertex quotient directly. The current code ignores sparse updates with zero net weight when it identifies affected endpoints; the archived campaign used the source commit recorded in its metadata.

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


## Table 11

For each seed from 1 to 5, the October 1 output contains five completed graphs and the October 2 supplement supplies CiteSeer. This gives 30 observations. An additional run at 21:40 on October 1 covered all six graphs for seed 1. On Patent it returned a final modularity of 0.540977, compared with 0.657468 in the original series. Substituting that run changes the five-seed sample standard deviation from 0.000382 to 0.052158. The archive does not establish which execution setting caused the difference.

The evidence ZIP includes the source records in `results/ieee-access-2026-2/`, the selection rule in `docs/s2cag-reproducibility.md`, and results for both choices on every graph under `stability` in `journal/ieee-access/analysis/validated_results.json`.
