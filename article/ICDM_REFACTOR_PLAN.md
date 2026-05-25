# ICDM 2026 Refactor Plan for ComNetX

Дата начала: 2026-05-25

Цель: довести статью о ComNetX до уровня сильной ICDM Research Track submission. Рабочий ориентир по шансам принятия после выполнения плана: 70-80% как целевой уровень убедительности, а не гарантия результата. Главное правило: не подставлять непроверенные цифры; все численные claims должны происходить из логов, воспроизводимых скриптов или явно помечаться как TODO.

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

Purpose: replace single-run tables with mean ± std.

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

Purpose: test drift.

Protocol:

- 50 and 100 mini-batches from the last chronological segment.
- Track Q, NMI, cumulative time, workload after each update.
- Add periodic-refresh baseline if local drift is visible: recompute every K in {10, 25, 50}.

### E6. Update-size/failure-mode stress test

Purpose: define operating envelope.

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

## 6. Iterative workflow

Iteration A: Paper structure and claims

- Make article internally consistent.
- Remove unsupported claims.
- Keep compact main tables and move raw details to appendix.

Iteration B: Config/script preparation

- Add scripts/paper/.
- Generate configs.
- User runs expensive measurements.

Iteration C: Result ingestion

- User places result JSONs under results/paper/.
- Run summarizers.
- Replace TBD values in article.

Iteration D: Reviewer-risk pass

- Check novelty framing.
- Check all claims have tables/figures.
- Check baselines are fair.
- Check known limitations are explicit but not self-defeating.

Iteration E: Final format pass

- IEEEtran compliance.
- Page budget.
- Captions self-contained.
- References verified.
- Appendix after references unless final template requires otherwise.

## 7. Definition of done

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

