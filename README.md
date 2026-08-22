# ComNetX ICDM 2026 Reproducibility Bundle

This branch is a reduced reviewer-facing bundle for the ICDM 2026 ComNetX
submission. It keeps the implementation, article source, figure scripts, and
canonical measurement data needed to inspect and reproduce the reported results.
Historical launch queues, legacy bundles, unrelated baselines, and heavy local
datasets are intentionally not tracked here.

The submitted ICDM manuscript remains frozen in `article/`. Journal revision
work is staged separately in `journal/ieee-access/` so that conference-specific
build scripts and the provenance of the existing measurements remain intact.

Historical anonymous repository link from the ICDM submission:
`https://anonymous.4open.science/r/comnetx-792B`

## What Is Included

- `article/`: submitted ICDM source, PDF, figures, and the baseline inclusion audit.
- `journal/ieee-access/`: clean workspace and curated revision notes for the
  IEEE Access manuscript and its evidence audit.
- `results/icdm-2026-1/`: canonical normalized measurement bundle used by the article.
- `src/`: ComNetX optimizer, launcher, metrics, dataset loading, and active baseline wrappers.
- `baselines/`: external code needed for the included empirical methods: DMoN,
  FLMIG, MAGI, MFC-TopoReg, PRGPT, and S2CAG.
- `scripts/paper/`: reproducibility utilities for ICDM configs, result packaging,
  figure regeneration, targeted cut metrics, workload profiling, and article checks.
- `.devcontainer/`, `Dockerfile`, `docker-compose.yml`: lightweight Python 3.10
  development container for syntax checks, smoke tests, unit tests, and figure regeneration.

## Quick Start

Start the dev container service:

```bash
docker compose up -d app
docker compose exec app bash
```

Inside the container:

```bash
make setup
make verify
```

`make verify` is deliberately lightweight. It compiles the reduced code tree,
runs smoke/unit tests, and regenerates the two article figures from
`results/icdm-2026-1`.

## Article And Measurements

The canonical measurement bundle is documented in:

```bash
results/icdm-2026-1/README.md
results/icdm-2026-1/manifest.json
```

Regenerate the paper figures from the committed measurement bundle:

```bash
python scripts/paper/plot_workload_speedup.py
python scripts/paper/plot_topology_ablation_pareto.py
```

Full article verification is available when LaTeX and Poppler tools are
installed:

```bash
python scripts/paper/verify_icdm_article.py
```

## Experiment Configs

Generate reviewer-facing experiment configs:

```bash
python scripts/paper/generate_icdm_configs.py
```

Run one generated config with explicit dataset paths:

```bash
python scripts/launch.py conf/paper_icdm/main_topology.json \
  --paths-config datasets-info/paths/astra.json
```

The committed dev container covers daily checks, not the full GPU research
stack. Full baseline reruns can require optional packages, GPU-specific wheels,
and the external datasets listed in the article.
