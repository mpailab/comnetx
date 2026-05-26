# ICDM 2026 Refactor Plan for ComNetX

Дата начала: 2026-05-25

Цель: довести статью о ComNetX до уровня сильной ICDM Research Track
submission. Рабочий ориентир: добиться такого качества постановки,
экспериментальной базы, анализа и подачи, чтобы внутренняя оценка шансов
принятия была около 80%. Это не гарантия результата, а целевой уровень
убедительности, под который должна строиться вся итеративная работа.

## 0. North Star and Operating Contract

This plan is the persistent memory for the ICDM submission effort. Every future
article edit, experiment script, result-registry update, and reviewer-risk pass
must be checked against this section first.

Core objective:

- Raise the paper to an ICDM-competitive evidence standard with an internal
  target acceptance likelihood of roughly 80%.
- Do not reduce the ambition of the paper merely because the current registry is
  incomplete. If the paper needs stronger evidence, design and run the missing
  measurements.
- Do not invent or silently approximate evidence. All numerical claims in the
  abstract, introduction, main tables, discussion, and conclusion must come from
  result logs, reproducible scripts, or explicitly marked placeholders awaiting
  a measurement round.

Iterative rule:

1. Start from the article's strongest ICDM-level story and identify which claims
   require evidence.
2. Check `results/registry/` first. If the registry already supports the claim,
   update the article and cite the exact table/figure source.
3. If evidence is missing or too weak, prepare a targeted measurement batch
   rather than weakening the article to fit incomplete data.
4. After the user runs measurements, rebuild `results/registry/`, analyze
   quality, runtime, variance, failures, and reviewer risk.
5. If the new data still leaves a serious weakness, prepare the next experiment
   round and repeat.
6. Stop iterating only when the Definition of Done below is satisfied and the
   remaining risks are acceptable for an 80%-target submission.

Important constraint: `999:10` is only a compatibility/smoke-like screen for
many backends. The main dynamic evidence must come from broader batch sweeps,
long-horizon runs, seed/split variance, workload analysis, DSBM stress tests,
and native-temporal LAGO comparisons.

Ten-day writing constraint:

- We have only 10 calendar days for the current article-writing sprint.
  Measurement planning must therefore be staged, not scheduled as one
  all-or-nothing wall-clock-critical batch.
- First ingest all currently available result JSONs into `results/registry/`.
  The registry is the decision point for every next run.
- Before launching week-long jobs, run short pilot batches that finish in hours
  or within one day. Use them to identify which claims, datasets, baselines, and
  update regimes actually strengthen the paper.
- Reserve the final 5-7 days for targeted long runs selected after registry
  analysis. Do not spend the whole remaining budget on broad exploratory sweeps.
- Treat the eight cn69 scripts as the maximal candidate suite. They are not a
  mandatory immediate launch plan; shorten, split, or skip them when the updated
  registry shows a better value/time tradeoff.
- Every long-running script must checkpoint results as soon as individual
  method/dataset/batch attempts complete. A stopped job must preserve completed
  measurements and enough manifest metadata to resume or redesign the run.

## 1. Что показывает анализ ICDM 2021-2025

Использованные источники для навигации по принятым работам:

- ICDM 2025 accepted papers: https://www3.cs.stonybrook.edu/~icdm2025/acceptedpapers.html
- ICDM 2024 accepted papers: https://icdm2024.org/accepted_papers/
- ICDM 2023 proceedings index: https://dblp.org/db/conf/icdm/icdm2023.html
- ICDM 2022 proceedings index: https://dblp.org/db/conf/icdm/icdm2022.html
- ICDM 2021 proceedings/accepted-paper indexes via DBLP and IEEE/ICDM pages
- ICDM 2026 CFP/formal constraints: https://icdm2026.neu.edu.cn/11666/list.htm

Наблюдаемые паттерны в близких темах:

1. Dynamic/temporal graph papers должны четко отделять задачу: snapshot stream, event stream, continuous-time temporal graph, fully dynamic graph, graph stream. Наша постановка должна называться batched snapshot maintenance, а не generic dynamic community detection.
2. Graph mining papers редко проходят только за счет speedup-table. Нужна причинная цепочка: why it works, which component matters, when it fails.
3. Для scalable methods ожидаются workload/memory/runtime breakdown, а не только total time.
4. Для GNN/feature-aware methods ожидаются feature ablations and seed variance. Наши random-feature local runs особенно уязвимы без ablation.
5. Для community/cluster papers важно не только modularity, но и external metrics, stability, sensitivity, and qualitative failure modes.
6. Accepted ICDM-style narrative обычно строится вокруг 3-5 research questions, compact main tables, detailed appendix/supplement, and reproducibility artifacts.

## 2. Новый главный тезис статьи

Слабая формулировка:

> ComNetX accelerates existing community detection methods by running them locally.

Сильная формулировка:

> ComNetX is a solver-agnostic dynamic reduction that converts batched snapshot updates into hierarchy-aware contracted local recomputation. Its key mechanism is affected-community closure: it preserves enough context for strong snapshot solvers to retain quality while shrinking the backend instance by orders of magnitude in update regimes where graph-radius expansion remains local.

Этот тезис должен проходить через abstract, introduction, method, experiments, discussion, conclusion.

## 3. Структура статьи после полного refactor

Main paper:

1. Abstract: task, gap, mechanism, strongest measured result, limits.
2. Introduction: dynamic graph mining trend, solver-specific dynamic methods vs solver-agnostic wrapper gap, contributions.
3. Problem and Interface: batched snapshot maintenance, black-box solver, metrics.
4. Method: affected set, radius expansion, hierarchical affected-community closure, contraction, feature aggregation, projection, complexity.
5. Experiments:
   - RQ1 quality/speed summary.
   - RQ2 comparison with native dynamic methods.
   - RQ3 locality/workload and memory explanation.
   - RQ4 ablation.
   - RQ5 robustness/failure modes.
6. Related Work: static modularity/GNN clustering, dynamic and local CD, continuous-time temporal communities, scalable graph mining.
7. Discussion/Limitations: exact operating envelope and fallback conditions.
8. Conclusion.
9. References.
10. Appendix: full per-dataset tables, extended configs, extra metrics.

## 4. Обязательные эксперименты для сильной версии

### E1. Main rerun with variance

Purpose: replace single-run tables with mean ± std and avoid presenting the
`999:10` run as the main dynamic evidence. Treat `999:10` as an all-backend
compatibility screen only.

Datasets:

- Primary: dyn_cora, dyn_acm, dyn_citeseer, patent, dyn_pubmed, arxivmath.
- Optional scale add-ons from datasets-info if runtime allows: arxivcs, dyn_ogbn-arxiv, arxivphy.

Methods:

- Core: Leidenalg naive/local, DF-Leiden dynamic/local, S2CAG naive/local, DMoN naive/local if feasible.
- Native temporal baseline: LAGO dynamic/local, reported separately because it
  optimizes continuous-time L-modularity rather than the same snapshot
  modularity objective.
- Secondary/appendix: FLMIG, PRGPT variants, MAGI, MFC.

Runs:

- 5 seeds or 5 stream splits.
- Report mean ± std for Q, NMI, total time.
- Use paired comparisons Local vs baseline on the same stream.

### E2. Ablation

Purpose: prove ComNetX is not a simple wrapper.

Minimum variants:

- Full ComNetX: r=1, L=3, closure on, contraction on, normalized feature aggregation.
- r=0, r=1, r=2.
- L=1, L=2, L=3, L=4.
- feature mode: dataset, random, onehot where feasible.
- aggregation mode: norm vs sum for feature-aware backends.

Datasets:

- dyn_cora or dyn_acm as small.
- dyn_pubmed as medium.
- arxivmath as large.

Backends:

- Leidenalg for topology-only stability.
- S2CAG or DMoN for feature-aware behavior.

Execution split:

- Use `leidenalg` for the full structural radius/depth grid. It is fast and isolates the topology-only ComNetX mechanism.
- Use a feature-aware backend only for feature-mode and feature-aggregation ablations. `leidenalg` cannot test these because it ignores node features.
- Keep GNN ablations light: run them first on `dyn_cora` and `dyn_pubmed`; add `arxivmath` only after the lightweight evidence is stable.

### E3. Locality/workload instrumentation

Already partially available:

- results/neighborhood/* gives graph-radius neighborhood growth.

Still missing:

- post-closure sizes |U_t^ell|;
- contracted backend nodes k_ell;
- contracted backend edges mbar_ell;
- per-level runtime.

Needed output:

- mean/max across updates;
- percentages relative to full graph;
- table linking B1, closure, contracted size, and speedup.

### E4. Runtime and memory breakdown

Purpose: defend scalability claim.

Measure:

- radius expansion time;
- closure time;
- aggregation time;
- backend call time;
- projection/cut time;
- CPU peak RSS;
- GPU peak allocation for GNN backends;
- conversion time separately.

### E5. Long-horizon robustness

Purpose: test drift. This is the real-data dynamic-stability protocol and must
carry the stronger runtime/quality claims, not the short `999:10` screen.

Protocol:

- 50 and 100 mini-batches from the last chronological segment for the broad
  batch-sensitivity sweep.
- 200 and 500 mini-batches on the larger real datasets for long-horizon drift.
- Run the same `m` values under `9:m`, `99:m`, and `999:m` when feasible, so
  the paper can separate the effect of initial-history size from the effect of
  update granularity.
- Track Q, NMI, cumulative time, workload after each update.
- Add periodic-refresh baseline if local drift is visible: recompute every K in {10, 25, 50}.

### E6. Update-size/failure-mode stress test

Purpose: define operating envelope.

Use the synthetic streams under `datasets-sbm/` for this block. They are not
`999:10` real-data streams: each file contains an initial graph at time `0`
followed by controlled update layers, and the suffixes `5_batches`,
`10_batches`, and `100_batches` represent different stress-test granularities.
This block should become the main evidence for random, hub-centered, and
community-internal update regimes.

Variants:

- random updates;
- hub-centered updates;
- community-internal updates;
- boundary-crossing updates if labels/communities are available.

Update sizes:

- 0.01%, 0.05%, 0.1%, 0.5%, 1% of edges.

Metrics:

- speedup;
- ΔQ against full recomputation;
- ΔNMI;
- closure size;
- max latency.

### E7. Synthetic SBM/temporal SBM

Purpose: controlled explanation of when ComNetX succeeds/fails.

Use:

- test/graphs/sbm_generator.py and/or a new paper script.

Scenarios:

- stable communities with sparse updates;
- gradual drift;
- abrupt community merge/split;
- hub update;
- boundary update.

## 5. Scripts to prepare

Create scripts under scripts/paper/.

1. generate_icdm_configs.py
   - emits JSON configs under conf/paper_icdm/.
   - configs for main, ablation, feature, long-horizon, and stress-test runs.

2. summarize_icdm_results.py
   - reads result JSON files from results/.
   - produces compact CSV/LaTeX tables:
     - main summary table;
     - full per-dataset table;
     - stability table;
     - ablation table.

3. summarize_neighborhood_table.py
   - reads results/neighborhood/*.json.
   - reproduces article Table 6 from raw logs.

4. trace_local_workload.py or runtime instrumentation patch
   - logs |B_r|, |U_t^ell|, k_ell, mbar_ell, and per-level timings.
   - preferred output: results/paper/workload_*.json.

5. generate_sbm_stress.py
   - generates temporal SBM scenarios for stress testing.
   - output should be loadable by Dataset or convertible to the existing dynamic format.

6. generate_cn69_measurement_scripts.py
   - emits the high-value cn69 configs under conf/paper_icdm/cn69/.
   - emits exactly eight GPU-oriented launch scripts under scripts/paper/cn69/.
   - these scripts are the current main measurement plan for closing the
     remaining ICDM evidence gaps.

## 5a. cn69 measurement package

Generate or refresh it with:

```bash
python3 scripts/paper/generate_cn69_measurement_scripts.py
```

Run the generated scripts inside the cn69 Docker containers, not through a
single `docker compose run app` job. The working directory inside every
container is `/home/dev/users/bokov/comnetx`. The containers are already bound
to individual GPUs, so the generated scripts must not set `CUDA_VISIBLE_DEVICES`
themselves. Shell logs are written to `output/` by default; standard result
JSON files remain under `results/`. Long measurements are always started in
background mode from the host with `docker exec -d`; do not use interactive
`docker exec -it` for the paper measurement batch.

Because the current sprint has a strict 10-day budget, do not launch the full
eight-script package blindly. The operating sequence is:

1. user places all already completed measurements under `results/`;
2. rebuild `results/registry/`;
3. inspect gaps, variance, failures, and time/quality tradeoffs;
4. run short pilots selected from the cn69 scripts or smaller configs;
5. only then schedule final multi-day jobs for the strongest missing evidence.

DSBM runs are especially interruption-sensitive. `run_dsbm_stress.py` must
write `results/paper_icdm/<run>.json`, `errors_<run>.json` when needed, and
`manifest_<run>.json` before the first algorithm call and after every completed
or failed attempt. Streams are ordered from smaller update budgets to larger
ones so early checkpoints contain usable evidence.

Current container pool:

- `dev_bokov`
- `dev_uporova`
- `dev_konovalov`
- `dev_egorov`
- `dev_egorov2`
- `dev_drobyshev`
- `dev_drobyshev2`
- `dev_drobyshev3`

Generated scripts:

- `scripts/paper/cn69/gpu0_real_topology_batch_sweep.sh`: Leiden/DF-Leiden
  real-data sensitivity over `9:10`, `9:50`, `9:100`, `99:10`, `99:50`,
  `99:100`, `999:10`, `999:50`, and `999:100`.
- `scripts/paper/cn69/gpu1_real_topology_long_horizon.sh`: 200/500-update
  long-horizon runs on `dyn_pubmed` and `arxivmath`.
- `scripts/paper/cn69/gpu2_s2cag_batch_sweep.sh`: S2CAG dataset features plus
  five random-feature seeds over `9:*`, `99:*`, and `999:*` with 50/100 updates.
- `scripts/paper/cn69/gpu3_dmon_batch_sweep.sh`: the DMoN counterpart to the
  S2CAG sweep.
- `scripts/paper/cn69/gpu4_feature_ablation_radius_aggregation.sh`: feature
  mode, radius, and aggregation ablation on representative attributed graphs.
- `scripts/paper/cn69/gpu5_lago_temporal_batch_sweep.sh`: native temporal
  LAGO, full-snapshot LAGO, and ComNetX-local LAGO.
- `scripts/paper/cn69/gpu6_dsbm_topology_stress.sh`: DSBM random,
  hub-centered, and community-internal stress streams for Leiden and DF-Leiden.
- `scripts/paper/cn69/gpu7_dsbm_lago_stress.sh`: the same DSBM stress suite
  for LAGO.

After the scripts finish, regenerate the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```

## 6. Current result registry

Canonical consolidated results registry:

- Directory: `results/registry/`.
- Generator script: `scripts/paper/collect_results_registry.py`.
- Usage: `python scripts/paper/collect_results_registry.py`.
- Registry README: `results/registry/README.md`.

When continuing the article refactor or replacing preliminary table values, look
here first:

- `results/registry/all_results.csv` and `all_results.json`: compact searchable
  records for all experiments and neighborhood analyses.
- `results/registry/all_results_with_series.json`: full records with per-update
  measurement series; use this for plots, long-horizon curves, and sanity checks.
- `results/registry/summary_by_run_key.csv`: grouped view by original algorithm
  string, dataset, batch strategy, ComNetX parameters, and backend settings; use
  this for mean/std tables and stability discussion.
- `results/registry/deduplicated_sources.csv`: exact or near-exact duplicate
  series that were collapsed; check this before treating repeated files as
  independent stability runs.
- `results/registry/errors.csv`: failed runs from `errors_*.json`; use this to
  explain missing baseline cells or decide what must be rerun.
- `results/registry/manifest.json`: latest counts and generated output paths.

Important convention: repeated runs with identical launch parameters are
preserved unless their full per-update metric/time series is exact or
near-identical after tolerance rounding. Therefore `summary_by_run_key.*` is the
starting point for paper tables, while `all_results_with_series.json` is the
source of truth for detailed curves and reproducibility checks.

Current LAGO status: the codebase and general configs contain the LAGO backend,
but the consolidated registry currently has zero `lago` result rows. Before
claiming anything about LAGO in the paper, run
`scripts/paper/cn69/gpu5_lago_temporal_batch_sweep.sh` and
`scripts/paper/cn69/gpu7_dsbm_lago_stress.sh`, then regenerate
`results/registry/`.

## 7. Iterative workflow

This workflow is cyclic, not one-pass. After each result batch, return to the
top-level objective: if the current evidence package is not yet strong enough
for the 80% target, prepare the next measurement round.

Iteration A: Paper structure and claims

- Make article internally consistent.
- Keep the strong target narrative, but mark unsupported numerical claims as
  placeholders until measured.
- Keep compact main tables and move raw details to appendix.

Iteration B: Config/script preparation

- Add scripts/paper/.
- Generate configs.
- User runs expensive measurements.
- Prefer targeted scripts that close concrete paper weaknesses over broad
  exploratory sweeps.
- Under the 10-day constraint, start with short pilots and postpone week-long
  runs until after the updated registry shows exactly which evidence is missing.

Iteration C: Result ingestion

- User places result JSONs under results/paper/.
- Run summarizers.
- Replace TBD values in article.
- If results are weak, diagnose whether the issue is a method limitation,
  parameter choice, dataset split, missing baseline, or missing statistic.
- When more evidence is needed, return to Iteration B instead of weakening the
  article prematurely.

Iteration D: Reviewer-risk pass

- Check novelty framing.
- Check all claims have tables/figures.
- Check baselines are fair.
- Check known limitations are explicit but not self-defeating.
- Re-estimate acceptance risk after every major result-ingestion round and list
  the remaining blockers.

Iteration E: Final format pass

- IEEEtran compliance.
- Page budget.
- Captions self-contained.
- References verified.
- Appendix after references unless final template requires otherwise.

## 8. Definition of done

The article is ready for serious ICDM submission only when:

- no result in abstract/introduction/conclusion depends on a single run;
- main table has mean ± std or is explicitly marked as current single-run preliminary;
- Table 6 has radius-neighborhood and contracted-workload evidence;
- ablation answers why hierarchy and closure matter;
- feature-mode ablation separates locality from random-feature effects;
- long-horizon plot shows no hidden drift or explains refresh fallback;
- memory table supports OOM/scalability claims;
- all related-work claims have checked references;
- no TODO/TBD remains in the main paper.
