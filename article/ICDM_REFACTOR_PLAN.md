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
long-horizon runs, seed/split variance, workload analysis, and DSBM stress
tests. Continuous-time temporal methods such as LAGO are related work only in
the current submission scope.

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
- Secondary/appendix: FLMIG, PRGPT variants, MAGI, MFC.

Runs:

- 5 seeds or 5 stream splits.
- Report mean ± std for Q, NMI, total time.
- Use paired comparisons Local vs baseline on the same stream.

LAGO scope rule:

- Do not spend the current experimental budget on LAGO comparisons. LAGO solves
  a related continuous-time link-stream problem, while this paper must stay
  focused on batched snapshot maintenance.
- Mention LAGO/Longitudinal Modularity in Related Work to show awareness of
  the temporal-community line and to clarify that it is complementary rather
  than a like-for-like baseline.
- Revisit LAGO only in a future extension with a genuine local-temporal adapter
  that extracts affected link streams across time. Until then, LAGO should not
  appear in the main experiment tables.

### E2. Ablation

Purpose: prove ComNetX is not a simple wrapper.

Minimum variants:

- Full ComNetX: r=1, L=3, closure on, contraction on, normalized feature aggregation.
- r=0, r=1, r=2.
- L=1, L=2, L=3, L=4.
- feature mode: dataset, random, onehot where feasible.
- aggregation mode: norm vs sum for feature-aware backends.
- closure/contraction variants: full, no_closure, no_contraction.

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
- Direct closure/no-contraction ablation is important for acceptance because it
  proves that ComNetX is not merely running the backend on a small induced
  subgraph. It is now measured through
  `scripts/paper/profile_smart_workload.py --variants full no_closure no_contraction`.
  Keep the main-paper table only if the pilot completes on both `dyn_pubmed`
  and `arxivmath`; otherwise move it to appendix or remove the table before
  final submission.

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
- mechanism figure linking contracted workload fraction to end-to-end speedup
  for the focused 3 x 2 method--dataset grid.

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
- Add a periodic-refresh baseline only if local drift is visible and there is
  time to implement it cleanly; otherwise discuss full refresh as the natural
  fallback regime without adding an unmeasured table row.

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
- `scripts/paper/cn69/gpu5_lago_temporal_batch_sweep.sh`: parked optional
  LAGO script from the earlier plan. Do not run it for the current main paper
  evidence chain.
- `scripts/paper/cn69/gpu6_dsbm_topology_stress.sh`: DSBM random,
  hub-centered, and community-internal stress streams for Leiden and DF-Leiden.
- `scripts/paper/cn69/gpu7_dsbm_lago_stress.sh`: parked optional LAGO stress
  script from the earlier plan. Do not run it for the current main paper
  evidence chain.

After the scripts finish, regenerate the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```

## 5b. Next short pilot batch: 2026-05-26

After ingesting the reorganized measurements under `results/paper_icdm/1`,
`results/paper_icdm/2`, and `results/paper_icdm/3`, the registry contains 2648
experiment records, 536 neighborhood records, 26 error records, 463
deduplicated sources, and no unreadable JSON files. The next batch must remain
a pilot batch: it should close high-value gaps and test feasibility before
scheduling any week-long final run.

Main registry observations:

- Topology sweeps are now broad enough for real-data batch sensitivity, and
  `results/paper_icdm/3` adds 200/500-update long-horizon rows for both
  `dyn_pubmed` and `arxivmath`. Additional repeated rows on the larger datasets
  are still useful for stability.
- The article still lacks the strongest scalability evidence: GPU memory,
  local workload size, contracted backend size, and timing breakdown for
  auxiliary transformations. This must be measured before the final paper pass.
- S2CAG dataset-feature high-history rows are now represented for the cn69
  batch sweep; the random-seed batch sweeps remain the main missing repeated
  GNN evidence.
- DMoN's dataset-feature sweep mostly deduplicates against earlier results; the
  random-seed/high-history rows remain the useful follow-up target.
- DSBM topology stress is now represented by the all-batches GPU6 run in
  `results/paper_icdm/3`; the next paper task is summarizing it into the
  update-size figure rather than rerunning the same stress script. LAGO evidence
  is no longer part of the main experimental plan; keep existing LAGO rows in
  the registry but do not use them in the main paper tables.

Generate the pilot package with:

```bash
python3 scripts/paper/generate_cn69_pilot_20260526.py
```

Generated locations:

- Configs: `conf/paper_icdm/cn69_pilot_20260526/`.
- Scripts: `scripts/paper/cn69_pilot_20260526/`.
- Runbook: `scripts/paper/cn69_pilot_20260526/README.md`.

The active pilot launches are:

- `gpu0_topology_variance_core.sh`: repeat key topology smart/dynamic rows on
  `dyn_pubmed` and `arxivmath`.
- `gpu1_workload_memory_profile.sh`: GPU-only profiling of affected vertices,
  post-closure sets, contracted backend sizes, auxiliary transformation time,
  backend time, CPU RSS, and CUDA peak memory.
- `gpu2_s2cag_acm_completion.sh`: finish the most informative missing S2CAG
  `dyn_acm` high-history rows without rerunning the full sweep.
- `gpu3_s2cag_pubmed_high_history.sh`: probe whether S2CAG becomes usable on
  `dyn_pubmed` when the initial history is large.
- `gpu4_dmon_pubmed_high_history.sh`: the DMoN counterpart for `dyn_pubmed`.
- `gpu5_dmon_feature_radius_cora.sh`: quick DMoN feature/radius check to see
  whether S2CAG feature-ablation conclusions generalize.
- `gpu6_dsbm_topology_micro.sh`: bounded DSBM topology pilot on the smallest
  update budget and all three update regimes.
- Parked legacy entry: `gpu7_lago_bridge_micro.sh`. Do not launch it for the
  current article campaign; use the GPU budget for workload, variance, GNN
  feasibility, or DSBM topology evidence instead.

Each active script has a default timeout and writes shell logs to `output/`.
DSBM and profiling scripts write checkpoint-style JSON files under
`results/paper_icdm/`.
After these pilots, rebuild `results/registry/` and decide whether to expand
only the winning directions into final multi-day runs.

## 5c. Mixed pilot/final batch: 2026-05-27

The article now keeps all acceptance-critical placeholders that have a runnable
measurement path. The direct closure/no-contraction ablation is restored because
it addresses the likely reviewer question: whether ComNetX is more than a small
induced-subgraph wrapper. It is measured by the workload profiler with
`--variants full no_closure no_contraction`.

Top-down article pass on 2026-05-27 fixed the main experimental visual set.
Do not add more main-paper tables or figures unless the registry reveals a
reviewer-critical gap that cannot be handled in prose or appendix. The main
figures are:

- method diagram: already present as `article/method.jpg`;
- quality--runtime Pareto summary: filled from `results/registry/` after the
  final registry rebuild, using the broad compatibility screen and focused
  repeated runs;
- workload-to-speedup mechanism plot: filled from `gpu1_workload_memory_pilot`
  and the consolidated registry;
- long-horizon curves: filled from the topology long-horizon run;
- DSBM update-size sensitivity: filled from the synthetic stress scripts.

Optional scale-out rows on `arxivcs`, `dyn_ogbn-arxiv`, or `arxivphy` are
appendix-only backup material. Do not spend the next week on them unless all
mandatory figures and tables above are already populated.

Generate the batch with:

```bash
python3 scripts/paper/generate_cn69_pilot_20260527.py
```

Generated locations:

- Configs: `conf/paper_icdm/cn69_pilot_20260527/`.
- Scripts: `scripts/paper/cn69_pilot_20260527/`.
- Runbook: `scripts/paper/cn69_pilot_20260527/README.md`.

This is an eight-launch package, but only the genuinely uncertain runs are
pilots:

- Pilot `gpu0_closure_contraction_pilot.sh`: short direct
  closure/contraction ablation on `dyn_pubmed` and `arxivmath`; fills or
  de-risks `tab:closure-ablation`.
- Pilot `gpu1_workload_memory_pilot.sh`: focused workload, timing, CPU RSS, and
  CUDA-memory profiling for the main 3 x 2 focused grid; fills
  `tab:contracted-workload`, `tab:breakdown`, and
  `fig:workload-speedup`. It profiles `999:10` plus a bounded prefix of
  `999:50`, so the mechanism plot is not based only on the short compatibility
  stream.
- Pilot `gpu2_dsbm_update_size_pilot.sh`: bounded synthetic update-size check
  on the smallest two update budgets; de-risks `fig:update-size`.
- Final `gpu3_topology_variance_final.sh`: repeated topology focused-grid
  measurements; fills `tab:stability` and contributes to
  `fig:quality-runtime-pareto`.
- Final `gpu4_topology_long_horizon_final.sh`: longer real-data topology
  horizons; fills `fig:long-horizon-curves`.
- Final `gpu5_s2cag_focused_final.sh`: focused S2CAG baseline/local rows.
  It contributes to `tab:stability`, `tab:breakdown`, and
  `fig:quality-runtime-pareto`.
- Conditional final `gpu6_closure_contraction_final.sh`: run after the gpu0
  pilot looks healthy; otherwise keep only the pilot result and remove or move
  the table before final submission.
- Final `gpu7_dsbm_update_size_final.sh`: larger DSBM update-size sweep with
  checkpointed JSON output; fills `fig:update-size`.

All launches are container-native background commands documented in the
runbook. Shell logs go to `output/`; checkpointed profiling/DSBM JSONs go to
`results/paper_icdm/`; ordinary launcher JSONs go to `results/`.
Monitor completion from the host with
`scripts/paper/cn69_pilot_20260527/monitor_cn69_jobs.sh`; use
`--watch 60 --tail 3` for a live one-minute dashboard with short log tails.

After ingesting `results/paper_icdm/3`, use the narrower eight-script follow-up
package instead of rerunning the now-completed broad cn69 sweeps:

```bash
python3 scripts/paper/generate_cn69_after3_20260527.py
```

Generated locations:

- Configs: `conf/paper_icdm/cn69_after3_20260527/`.
- Scripts: `scripts/paper/cn69_after3_20260527/`.
- Runbook: `scripts/paper/cn69_after3_20260527/README.md`.

The after-3 package spends the core eight GPU slots on workload/profile
evidence, direct closure/contraction ablation, and S2CAG/DMoN random-seed
high-history rows. Optional `extra*` scripts add focused closure/contraction,
radius, and feature-mode checks after containers free up. The package
deliberately avoids repeating the topology, DSBM, LAGO, S2CAG dataset-feature,
and feature-ablation sweeps already represented in the registry.
Operational note after the extra24/extra25 runs: DF-Leiden `no_contraction` on
`arxivmath` hit cuSPARSE insufficient-resource failures for both `9:500` and
`9:100`. Treat these as contraction-necessity/operating-envelope evidence and
do not schedule more DF-Leiden `no_contraction` arxivmath follow-ups without an
implementation or hardware change.

Required small-control block after this pass: add `dyn_cora` only after
workload-profile measurements exist, not from ordinary launcher rows. The
script `scripts/paper/cn69_after3_20260527/extra27_dyn_cora_small_control.sh`
profiles `dyn_cora` on `999:50` for:

- Leidenalg and DF-Leiden full ComNetX, to fill the small-graph rows for
  `tab:contracted-workload`, `fig:workload-speedup`, and `tab:breakdown`;
- S2CAG random-feature full ComNetX, to fill the feature-aware small-graph row
  in the same workload/profile artifacts;
- Leidenalg `full`, `no_closure`, and `no_contraction`, to fill the
  small-graph control rows in `tab:closure-ablation`.

Do not insert `dyn_cora` into those main-paper workload/profile artifacts until
the `extra27` JSONs have been ingested into `results/registry/`. Existing
ordinary experiment rows are sufficient for appendix and broad compatibility
screens, but not for contracted-size, memory, or timing-breakdown claims.

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
and the consolidated registry has pilot `lago` rows from the
`lago_temporal_batch_sweep_20260526_1339.json` files under
`results/paper_icdm/2` and `results/paper_icdm/3`. These rows are preserved in
the registry for traceability but are not part of the current submission story.
The article should cite LAGO only in Related Work as a continuous-time
temporal-community method that is complementary to ComNetX's batched snapshot
maintenance setting.

## 6a. Table and baseline selection policy

When updating the article, keep the tables aligned with the paper's evidence
chain rather than filling rows opportunistically:

- Breadth screen: `tab:main-results` and Appendix A use all six real datasets
  and every implemented baseline with comparable output. This is the only place
  where FLMIG, PRGPT, MAGI, DMoN, and MFC need to appear by default unless a
  focused reviewer claim requires a complete additional matrix.
- Focused real-data tables: use the complete 3 x 2 grid
  `{Leidenalg, DF-Leiden, S2CAG} x {dyn_pubmed, arxivmath}` whenever the table
  is about direct method comparison, statistical robustness, contracted
  workload, or runtime/memory breakdown. Avoid singleton rows even when a
  single row is technically interesting; move such evidence to prose,
  appendix, or a dedicated table.
- Small-graph control: `dyn_cora` is the preferred small graph if the paper
  needs a small/medium/large mechanism view. It may be added to
  `tab:contracted-workload`, `fig:workload-speedup`, `tab:closure-ablation`,
  and `tab:breakdown` only after the required `extra27` workload-profile block
  has been run and ingested. Do not add `dyn_acm`, `dyn_citeseer`, or `patent`
  to those focused artifacts unless a later reviewer-risk pass identifies a
  stronger reason than the small-control role.
- Topology/locality claim: Leidenalg on `dyn_pubmed` and `arxivmath` tests the
  strongest topology-only full-recomputation baseline in medium/large regimes
  where local contraction should matter.
- Native dynamic claim: DF-Leiden on the same two datasets tests where native
  dynamic topology methods are already fast (`dyn_pubmed`) and where local
  restriction becomes valuable (`arxivmath`).
- Feature-aware/GPU claim: S2CAG on the same two datasets connects locality,
  memory pressure, and GPU execution.
- DMoN/MAGI/MFC large-graph rows should be used for feasibility, memory, or
  OOM-avoidance discussion only after a complete table-specific comparison is
  available; do not insert them as lone focused-table rows.
- DSBM streams belong only to the operating-envelope/failure-mode block. Do not
  mix synthetic DSBM rows into real-data quality tables.
- LAGO remains Related Work only unless we build a genuine continuous-time
  local adapter and a like-for-like temporal-community evaluation.

Placeholder rule for the draft:

- Keep TBD placeholders only for measurements that already have a runnable
  path in the repository: workload/profile columns, runtime/memory breakdown,
  closure/contraction variants, quality--runtime Pareto, workload-to-speedup
  mechanism plot, long-horizon curves, and DSBM update-size sensitivity.
- Remove incomplete main-paper tables that require new algorithmic variants or
  a new evaluation protocol unless we explicitly schedule and implement those
  measurements.

Frozen main-paper evidence package after the 2026-05-27 pass:

- `tab:main-results` plus `fig:quality-runtime-pareto`: breadth and
  quality--runtime frontier.
- `tab:stability`: repeated focused-grid robustness.
- `tab:workload`, `tab:contracted-workload`, `fig:workload-speedup`, and
  `tab:breakdown`: mechanism, contracted workload, runtime decomposition, and
  memory.
- `tab:ablation`, `tab:feature-ablation`, and `tab:closure-ablation`: component
  necessity.
- `tab:long-horizon` plus `fig:long-horizon-curves`: drift and cumulative
  runtime under longer streams.
- `tab:dsbm` plus `fig:update-size`: operating envelope and failure modes.

This set is sufficient for the planned ICDM story. Further additions should go
to appendix unless they replace a weak mandatory artifact.

2026-05-28 measurement queue: do not weaken the article before the full
measurement picture is available. The next prepared package is
`scripts/paper/cn69_final_20260528/`, generated by
`scripts/paper/generate_cn69_final_20260528.py`.

Registry check before scheduling:

- Already covered: PubMed/Arxiv workload-profile rows, direct
  closure/contraction rows, S2CAG/DMoN random-seed follow-ups from
  `results/paper_icdm/4`. Do not repeat these cells.
- Still required for main-paper artifacts: the `dyn_cora` small-control
  workload-profile block for `tab:contracted-workload`,
  `fig:workload-speedup`, `tab:closure-ablation`, and `tab:breakdown`.
- Still required for long-horizon completeness: newer cn69 rows already provide
  NMI for the Leiden and DF-Leiden `dyn_pubmed` `999:100` cells, but two
  LD-Leiden rows still have runtime/modularity without final NMI. The new GPU0
  script reruns only those two LD-Leiden rows with ground-truth metrics
  enabled.
- Still required for `fig:update-size`: random DSBM solver measurements are
  absent, and hub-centered DSBM is absent except for the already ingested
  `hubs/mc1450/100_batches/leidenalg-naive` row. The DSBM runner now supports
  `--skip-registry` to avoid repeating ingested DSBM cells.

Core eight for the 2026-05-28 queue:

- GPU0: missing LD-Leiden long-horizon NMI plus random DSBM 5/10-batch slices.
- GPU1: required `dyn_cora` small-control profile plus hub DSBM 5/10-batch
  slices.
- GPU2/GPU4/GPU6: random DSBM 100-batch slices split by update size.
- GPU3/GPU5/GPU7: hub DSBM 100-batch slices split by update size.

Bad-locality graphs (`dyn_blogcatalog`, `dyn_wikics`, `brain`) are not part of
the core queue because they do not currently close a numbered main-paper table
or figure. Keep them in the 24 optional `extra*.sh` follow-ups as admissibility
and operating-envelope probes if core containers finish early.

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
- quality--runtime Pareto figure shows the frontier across supported backends
  and datasets;
- Table 6 has radius-neighborhood and contracted-workload evidence;
- ablation answers why radius, hierarchy depth, feature handling, closure, and
  contraction matter;
- workload/profile evidence quantifies post-closure and contracted instance sizes;
- workload-to-speedup figure links the measured contraction mechanism to
  observed runtime gains;
- feature-mode ablation separates locality from random-feature effects;
- long-horizon plot shows no hidden drift across the measured streams;
- memory table supports OOM/scalability claims;
- all related-work claims have checked references;
- no TODO/TBD remains in the main paper.
