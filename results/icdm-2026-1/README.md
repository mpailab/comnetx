# icdm-2026-1 Measurement Bundle

This directory stores the canonical measurement data used by the ICDM 2026 article.

Measurements are split by data type. Experiment time series, workload
profiles, neighborhood-growth rows, and cut metrics each have a
separate JSON file with full payloads. Original result file paths are
not embedded in records.

## Article Navigation

- Compatibility study (tab:main-results): `measurements/experiment_measurements.json` (`measurement_family`: `single_container_real_graph`, `article_compatibility_baselines`)
- Repeated 999:10 robustness (tab:stability): `measurements/experiment_measurements.json` (`measurement_family`: `dyn_pubmed_leiden_repeats`, `single_container_real_graph`)
- Empirical neighborhood growth (tab:workload): `measurements/neighborhood_measurements.json` (`measurement_family`: `article_neighborhood_growth`)
- Contracted workload and speedup (fig:workload-speedup): `measurements/workload_profiles.json` (`measurement_family`: `article_completion_workload_profiles`, `single_container_workload_profiles`). Speed ratios use paired records from measurements/experiment_measurements.json.
- Leiden topology ablation (fig:topology-ablation): `measurements/experiment_measurements.json` (`measurement_family`: `article_completion_real_graph`, `single_container_real_graph`)
- Gamma-resolution sensitivity: `measurements/experiment_measurements.json` (`measurement_family`: `gamma_sweep`)
- S2CAG feature-mode ablation (tab:feature-ablation): `measurements/experiment_measurements.json` (`measurement_family`: `single_container_real_graph`)
- Leiden closure/contraction ablation (tab:closure-ablation): `measurements/workload_profiles.json` (`measurement_family`: `single_container_workload_profiles`)
- Long-horizon topology endpoints (tab:long-horizon): `measurements/experiment_measurements.json` (`measurement_family`: `single_container_real_graph`)
- Directed-control reviewer check: `measurements/experiment_measurements.json` (`measurement_family`: `single_container_directed_control`)
- Conductance / normalized-cut reviewer check: `measurements/cut_metrics.json` (`measurement_family`: `leiden_cut_metrics`)
- Controlled DSBM stress test: `measurements/experiment_measurements.json` (`measurement_family`: `single_container_dsbm`)

## Files

- `manifest.json`: navigation from article items to data files.
- `measurements/experiment_measurements.json`: experiment records with complete update series.
- `measurements/workload_profiles.json`: workload-profile records with complete profile payloads.
- `measurements/neighborhood_measurements.json`: neighborhood growth rows used by the empirical neighborhood table.
- `measurements/cut_metrics.json`: targeted structural-quality metrics.

## Record Counts

- Total unified records: `480`

## Records By Type

- `cut_metrics`: `4` records.
- `experiment`: `446` records.
- `neighborhood`: `6` records.
- `workload_profile`: `24` records.

## Records By Family

- `article_compatibility_baselines`: `223` records.
- `article_completion_real_graph`: `11` records.
- `article_completion_workload_profiles`: `9` records.
- `article_neighborhood_growth`: `6` records.
- `dyn_pubmed_leiden_repeats`: `10` records.
- `gamma_sweep`: `12` records.
- `leiden_cut_metrics`: `4` records.
- `single_container_directed_control`: `4` records.
- `single_container_dsbm`: `60` records.
- `single_container_real_graph`: `126` records.
- `single_container_workload_profiles`: `15` records.
