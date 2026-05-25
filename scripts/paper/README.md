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

## Summarize result JSONs

Flatten one or more result files:

```bash
python3 scripts/paper/summarize_icdm_results.py results/paper/run.json --output flat.csv
```

Aggregate by algorithm, dataset, and batch:

```bash
python3 scripts/paper/summarize_icdm_results.py results/paper/*.json --summary --output summary.csv
```

## Rebuild the neighborhood table

```bash
python3 scripts/paper/summarize_neighborhood_table.py
```

This reproduces the LaTeX rows for the current Table 6 from
`results/neighborhood/*.json`.

