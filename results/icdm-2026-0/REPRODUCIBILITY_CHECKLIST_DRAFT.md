# ICDM 2026 Reproducibility Checklist Draft

This draft is intended for completing the ICDM submission checklist. It is
anonymized and refers only to bundled measurement artifacts.

## Problem And Scope

- Task: batched snapshot maintenance for dynamic community detection.
- Method: ComNetX, a solver-agnostic local hierarchical adapter around existing
  community detection backends.
- Main limitation: the local update is effective when affected communities
  remain small; large, hub-heavy, or poorly localized updates can require a full
  refresh.

## Data And Measurements

- Real graph measurements are stored in
  `results/icdm-2026-0/measurements/real_graph_measurements.json`.
- Neighborhood-size measurements are stored in
  `results/icdm-2026-0/measurements/neighborhood_measurements.json`.
- Workload-profile measurements are stored in
  `results/icdm-2026-0/measurements/workload_profiles.json`.
- Synthetic dynamic stochastic block model measurements are stored in
  `results/icdm-2026-0/measurements/synthetic_dsbm_measurements.json`.
- The bundle manifest is `results/icdm-2026-0/manifest.json`; it records 4017
  measurement entries and includes internal consistency checks.

## Experimental Protocol

- The broad compatibility study uses the common `999:10` stream protocol.
- Long-horizon topology rows use the 500-update `9:500` records for
  `dyn_pubmed` and `arxivmath`.
- The S2CAG feature-mode ablation table uses `999:100` records.
- The DSBM stress table reports per-stream mean affected-vertex percentages
  over the generated 5-, 10-, and 100-batch granularities.
- Reported tables exclude failed or out-of-memory baseline entries when
  computing finite aggregate speedups, and the paper states this convention.

## Code And Figure Generation

- Main measurement utilities are documented in `scripts/paper/README.md`.
- The paper figures are regenerated with:

```bash
python3 scripts/paper/plot_workload_speedup.py
python3 scripts/paper/plot_topology_ablation_pareto.py
```

- The scripts write `article/workload_speedup.pdf` and
  `article/topology_ablation_pareto.pdf`, which are included by
  `article/article.tex`.

## Randomness And Repetitions

- Repeated non-random timing measurements are reported separately in the
  robustness table.
- The paper avoids claiming statistical significance for the broad
  compatibility table; it is presented as breadth-oriented evidence.
- Feature-aware local GNN runs change both locality and feature mode; the paper
  explicitly treats those rows as evidence for the complete deployment recipe,
  not as an isolated feature-causality claim.

## Resource Reporting

- The article reports cumulative processing time and out-of-memory avoidance in
  the main compatibility table.
- Workload-profile records include profiled maintenance costs, contracted
  backend sizes, and memory-related fields.
- Timing measurements were performed on a Dockerized Linux server with two AMD
  EPYC 7742 64-core CPUs, 2.0 TiB RAM, and eight NVIDIA A100-SXM4 GPUs with
  80 GB memory each. The NVIDIA driver was 535.216.03, `nvidia-smi` reported
  CUDA 12.2, and the container used Python 3.10.13, PyTorch 2.3.1 with CUDA
  11.8, and TensorFlow 2.14.0.
- Hardware metadata should be collected on the same server used for timing
  measurements with the anonymized paper utility before final submission. The
  output includes CPU, memory, GPU, driver, CUDA, Python, and package versions
  while omitting hostnames and user paths. The paper-format output is intended
  for the compact hardware sentence in the experimental setup.
- Insert the hardware sentence with
  `scripts/paper/insert_hardware_sentence.py` after collecting the JSON on the
  measurement server, then run `scripts/paper/verify_icdm_article.py --final`.
  The insertion helper rejects local no-GPU container summaries.
