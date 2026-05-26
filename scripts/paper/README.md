# Paper Measurement Scripts

These helpers support the ICDM paper iteration. They do not run experiments by
themselves unless explicitly called.

## Generate configs

```bash
python3 scripts/paper/generate_icdm_configs.py
```

This writes JSON configs to `conf/paper_icdm/`. Run any generated config with
the existing launcher:

```bash
python3 scripts/launch.py conf/paper_icdm/main_topology.json
```

Use `--include-scale` if optional larger datasets should be added to the main
configs.

The configs are split by paper claim:

- `main_topology.json`: the main fast topology-only comparison.
- `ablation_topology_radius_depth.json`: structural ablation with `leidenalg`.
- `main_gnn_feature.json`: feature-aware feasibility and quality comparison.
- `ablation_feature_modes.json`: dataset/random/onehot feature ablation.
- `ablation_feature_aggregation.json`: `norm` vs `sum` feature aggregation.
- `ablation_gnn_radius_light.json`: small GNN-only radius check.
- `long_horizon_topology.json`: long-horizon topology robustness.

The feature ablations intentionally use feature-aware backends only. `leidenalg`
is used for the full radius/depth ablation because it is fast and topology-only;
it cannot test feature modes or feature aggregation.

## Summarize result JSONs

Flatten one or more result files:

```bash
python3 scripts/paper/summarize_icdm_results.py results/paper/run.json --output flat.csv
```

Aggregate by algorithm, dataset, and batch:

```bash
python3 scripts/paper/summarize_icdm_results.py results/paper/*.json --summary --output summary.csv
```

## Build the full results registry

```bash
python3 scripts/paper/collect_results_registry.py
```

This scans `results/**/*.json` and writes a consolidated, searchable registry to
`results/registry/`. It preserves repeated runs with the same parameters for
stability analysis, while collapsing exact or near-exact duplicate series into
`deduplicated_sources.*`.

## Rebuild the neighborhood table

```bash
python3 scripts/paper/summarize_neighborhood_table.py
```

This reproduces the LaTeX rows for the current Table 6 from
`results/neighborhood/*.json`.
