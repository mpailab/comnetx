# Paper Measurement Utilities

This directory keeps reusable tooling for paper measurements and result
packaging. Historical machine-specific launch queues should live in git history
and generated measurement bundles, not as active scripts in this directory.

## Generate Baseline Configs

```bash
python3 scripts/paper/generate_icdm_configs.py
```

Writes reusable ICDM experiment configs under `conf/paper_icdm/` on demand. The
generated config directory is intentionally not kept as active source. Run a
generated config with the standard launcher:

```bash
python3 scripts/launch.py conf/paper_icdm/main_topology.json
```

Use `--include-scale` to add optional larger datasets to the generated configs.

## Run Measurements

Profile smart-mode workload, timing breakdown, and memory:

```bash
python3 scripts/paper/profile_smart_workload.py --help
```

Run DSBM stress measurements over synthetic streams:

```bash
python3 scripts/paper/run_dsbm_stress.py --help
python3 scripts/paper/run_dsbm_stress.py --all-batches --list-streams
```

Summarize available DSBM stream metadata:

```bash
python3 scripts/paper/summarize_dsbm_streams.py
```

## Process Results

Build a consolidated registry from raw result JSON files:

```bash
python3 scripts/paper/collect_results_registry.py
```

Flatten or aggregate launcher result JSON files:

```bash
python3 scripts/paper/summarize_icdm_results.py results/run.json --output flat.csv
python3 scripts/paper/summarize_icdm_results.py results/*.json --summary --output summary.csv
```

Import launcher stdout summaries when only text logs are available:

```bash
python3 scripts/paper/import_text_measurement_logs.py smart.txt experiment.txt leidenalg.txt --stamp 20260526
```

Build a self-contained supplemental measurement bundle:

```bash
python3 scripts/paper/package_icdm_results.py --tag icdm-2026-0 --force
```
