# ICDM Paper Utilities

This directory keeps the reusable utilities needed by the reduced ICDM reviewer
bundle. Historical machine-specific launch queues live in git history and are
not active source in this branch.

## Generate Experiment Configs

```bash
python3 scripts/paper/generate_icdm_configs.py
```

Writes reusable ICDM experiment configs under `conf/paper_icdm/` on demand. Run
a generated config with the standard launcher:

```bash
python3 scripts/launch.py conf/paper_icdm/main_topology.json --paths-config datasets-info/paths/astra.json
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

Collect anonymized hardware metadata on the same server used for timing
measurements:

```bash
python3 scripts/paper/collect_hardware_info.py --output tmp/hardware_summary.json
python3 scripts/paper/collect_hardware_info.py --format text
python3 scripts/paper/collect_hardware_info.py --format paper
```

Insert the paper-ready hardware sentence only after the JSON was collected on
the measurement server:

```bash
python3 scripts/paper/insert_hardware_sentence.py --hardware-json tmp/hardware_summary.json --dry-run
python3 scripts/paper/insert_hardware_sentence.py --hardware-json tmp/hardware_summary.json
```

The insertion helper refuses no-GPU local-container summaries and requires
NVIDIA/CUDA details, because the article reports GPU-enabled baselines.

## Plot Paper Figures

Regenerate the main-paper figures:

```bash
python3 scripts/paper/plot_workload_speedup.py
python3 scripts/paper/plot_topology_ablation_pareto.py
```

The scripts write `article/workload_speedup.pdf` and
`article/topology_ablation_pareto.pdf`, which are included by the ICDM paper
source.

## Verify Article

Run the ICDM article hygiene checks:

```bash
python3 scripts/paper/verify_icdm_article.py
```

The verifier rebuilds the PDF, checks the 10-page limit, scans the source and
PDF text for internal markers, validates labels and citations, checks anonymity
metadata, rejects appendices in the main source, and requires at least 50 cited
references.

After collecting and inserting the hardware sentence from the measurement
server, run the final-submission gate:

```bash
python3 scripts/paper/verify_icdm_article.py --final
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

The committed `results/icdm-2026-1` directory is the canonical supplemental
measurement bundle. When the raw or predecessor measurement inputs are present,
rebuild it with:

```bash
python3 scripts/paper/package_icdm_2026_1_results.py
```
