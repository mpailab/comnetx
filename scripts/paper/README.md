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
- `main_lago_temporal.json` and `scale_lago_temporal.json`: parked optional
  LAGO configs from an earlier plan. Do not run them for the current main paper
  evidence chain; LAGO is now handled in Related Work only.
- `ablation_feature_modes.json`: dataset/random/onehot feature ablation.
- `ablation_feature_aggregation.json`: `norm` vs `sum` feature aggregation.
- `ablation_gnn_radius_light.json`: small GNN-only radius check.
- `long_horizon_topology.json`: long-horizon topology robustness.

The feature ablations intentionally use feature-aware backends only. `leidenalg`
is used for the full radius/depth ablation because it is fast and topology-only;
it cannot test feature modes or feature aggregation.

## Generate the cn69 measurement package

```bash
python3 scripts/paper/generate_cn69_measurement_scripts.py
```

This writes the high-value measurement configs to `conf/paper_icdm/cn69/` and
the eight GPU-oriented launch scripts to `scripts/paper/cn69/`. The cn69 package
is the main follow-up protocol for the ICDM revision: it replaces a single
`999:10` dependence with sweeps over `9:*`, `99:*`, and `999:*` initial
fractions, 50/100/200/500 update horizons, random-feature seed repeats, and
controlled DSBM stress streams. LAGO scripts remain in the tree for
traceability but are parked for the current submission scope.

With the current 10-day paper sprint, treat these scripts as a candidate suite,
not as an immediate all-at-once schedule. Ingest the latest available results
first, rebuild `results/registry/`, run short pilots that finish in hours or
within one day, and schedule final multi-day runs only after the registry shows
which evidence is still missing.

Run the generated scripts inside the GPU-bound cn69 Docker containers from
`/home/dev/users/bokov/comnetx`. The scripts do not set `CUDA_VISIBLE_DEVICES`;
the container binding selects the GPU. Start long measurements in background
mode with `docker exec -d`. Shell logs are written to `output/`.

```bash
scripts/paper/cn69/gpu0_real_topology_batch_sweep.sh
scripts/paper/cn69/gpu1_real_topology_long_horizon.sh
scripts/paper/cn69/gpu2_s2cag_batch_sweep.sh
scripts/paper/cn69/gpu3_dmon_batch_sweep.sh
scripts/paper/cn69/gpu4_feature_ablation_radius_aggregation.sh
scripts/paper/cn69/gpu5_lago_temporal_batch_sweep.sh  # parked; do not run for main paper
scripts/paper/cn69/gpu6_dsbm_topology_stress.sh
scripts/paper/cn69/gpu7_dsbm_lago_stress.sh          # parked; do not run for main paper
```

After the jobs finish, rebuild `results/registry/` with
`python3 scripts/paper/collect_results_registry.py`.

## Generate the next cn69 pilot batch

```bash
python3 scripts/paper/generate_cn69_pilot_20260526.py
```

This writes a narrower eight-launch pilot package to
`conf/paper_icdm/cn69_pilot_20260526/` and
`scripts/paper/cn69_pilot_20260526/`. Use it after the current registry has
been rebuilt from `results/paper_icdm/1` and `results/paper_icdm/2`. The pilot
batch targets GPU workload/memory breakdown, incomplete S2CAG/DMoN
high-history rows, repeated topology stability, and bounded DSBM checks.
The old LAGO pilot is parked; use available GPU time for the core evidence
chain instead.

Generate the 2026-05-27 mixed pilot/final batch:

```bash
python3 scripts/paper/generate_cn69_pilot_20260527.py
```

This writes `conf/paper_icdm/cn69_pilot_20260527/` and
`scripts/paper/cn69_pilot_20260527/`. The batch restores the direct
closure/no-contraction ablation through `profile_smart_workload.py --variants`
and uses the remaining GPU slots for focused topology, S2CAG, workload/memory,
and DSBM measurements. See
`scripts/paper/cn69_pilot_20260527/README.md` for background Docker launch
lines.

After ingesting `results/paper_icdm/3`, generate the narrower follow-up batch
that avoids completed cn69 sweeps. The core eight scripts use the GPU slots for
workload/profile and GNN seed measurements; the `extra*` scripts add optional
focused closure/contraction, radius, and feature-mode checks after any
containers free up:

```bash
python3 scripts/paper/generate_cn69_after3_20260527.py
```

This writes `conf/paper_icdm/cn69_after3_20260527/` and
`scripts/paper/cn69_after3_20260527/`.

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

## Import text measurement logs

If only launcher stdout summaries are available, convert them into ordinary
result JSON files before rebuilding the registry:

```bash
python3 scripts/paper/import_text_measurement_logs.py smart.txt experiment.txt leidenalg.txt --stamp 20260526
python3 scripts/paper/collect_results_registry.py
```

The importer writes to `results/imported_logs/`. It skips rows that are already
near-identical to records in `results/registry/`, but keeps repeated runs with
the same launch parameters when their measured values differ.

## Summarize and run DSBM stress streams

The synthetic DSBM datasets in `datasets-sbm/` do not use the real-data
`999:10` protocol. Each stream already contains an initial graph at time `0`
and controlled update layers after it. The available suffixes such as
`5_batches`, `10_batches`, and `100_batches` should be treated as independent
stress-test granularities.

Summarize all generated DSBM streams:

```bash
python3 scripts/paper/summarize_dsbm_streams.py
```

Run one explicit stress-test granularity:

```bash
python3 scripts/paper/run_dsbm_stress.py --batch-suffix 10_batches --methods leidenalg dfleiden --modes naive smart dynamic
```

The DSBM runner now fails loudly if no streams are selected or if every
algorithm run fails under `--catch-errors`. It also writes
`manifest_<run>.json` next to the result file with selected stream counts,
attempted runs, successful runs, error counts, and the current attempt. It
creates a checkpoint before the first algorithm call and after each successful
or failed attempt, using atomic JSON replacement so completed measurements
survive interrupted jobs. Streams are ordered from smaller update budgets to
larger ones so early checkpoints are useful. Before a long cn69 run, check
stream discovery:

```bash
python3 scripts/paper/run_dsbm_stress.py --root "${DSBM_ROOT:-datasets-sbm}" --all-batches --list-streams
```

## Rebuild the neighborhood table

```bash
python3 scripts/paper/summarize_neighborhood_table.py
```

This reproduces the LaTeX rows for the current Table 6 from
`results/neighborhood/*.json`.
