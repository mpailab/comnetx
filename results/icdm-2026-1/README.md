# icdm-2026-1 Measurement Bundle

This directory is a compact supplemental measurement snapshot for the ICDM 2026 reviewer-followup and single-container rerun pass.
It preserves the earlier follow-up records and adds the undirected real-graph outputs from `results/paper_icdm_single_container_rerun/series_14`.
DSBM and directed-control measurements are intentionally outside this bundle update.

## Provenance

- Tag: `icdm-2026-1`
- Reference commit: `c0121d6413484f691bc52ba5545fd5aba531691c`
- Source roots:
  - `results/paper_icdm_reviewer_followup/series_12`
  - `results/paper_icdm_single_container_rerun/series_14`
- Source JSON files: `33`
- Total measurement entries: `163`

## Measurement Entries By Type

- `experiment`: `148` entries.
- `workload_profile`: `15` entries.

## Measurement Entries By Category

- `gamma_sweep_measurements`: Reviewer follow-up Leiden gamma sensitivity measurements for dyn_pubmed and arxivmath. Entries: `12`.
- `dyn_pubmed_leiden_repeats`: Five single-container repeated Leiden measurements on dyn_pubmed, used to audit runtime variation. Entries: `10`.
- `single_container_real_graph_measurements`: Single-container rerun experiment measurements for undirected real-graph ICDM paper runs: Leidenalg, DF-Leiden, S2CAG, feature ablations, topology ablations, and long-horizon endpoints. DSBM and directed-control measurements are intentionally excluded. Entries: `126`.
- `single_container_workload_profiles`: Single-container rerun workload-profile measurements for Leidenalg, DF-Leiden, and S2CAG locality/closure analysis. Entries: `15`.
