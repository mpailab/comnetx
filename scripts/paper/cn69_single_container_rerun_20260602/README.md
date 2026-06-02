# Single-container rerun queue

This queue remeasures article-critical runtime data in one container to avoid
CPU memory contention from concurrent eight-container runs.

Launch from the server:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=14 scripts/paper/cn69_single_container_rerun_20260602/run_single_container_rerun.sh'
```

The runner is sequential and writes every task into
`results/paper_icdm_single_container_rerun/series_${PAPER_ICDM_SERIES}`. It
creates `.done` markers per task/repetition and can be restarted; completed
tasks are skipped. `scripts/launch.py`, `profile_smart_workload.py`, and
`run_dsbm_stress.py` all checkpoint JSON after completed inner runs, so partial
results survive interruption.

Order and rationale:

1. `Leiden 999:10 Table II small/medium graphs`, 1 repeat: `dyn_cora`, `dyn_acm`, `dyn_citeseer`, and `patent`.
2. `DF-Leiden 999:10 Table II small/medium graphs`, 1 repeat: the same four Table II graphs.
3. `S2CAG dataset-feature 999:10 Table II small/medium graphs`, 1 repeat: the same four Table II graphs.
4. `Leiden 999:10 core stability`, 5 repeats: `dyn_pubmed` and `arxivmath`.
5. `DF-Leiden 999:10 core stability`, 5 repeats: `dyn_pubmed` and `arxivmath`.
6. `S2CAG dataset-feature 999:10 core stability`, 5 repeats: `dyn_pubmed` and `arxivmath`.
7. `Leiden 9:500 long horizon`, 1 repeat: long-horizon table.
8. `DF-Leiden 9:500 long horizon`, 1 repeat: long-horizon native dynamic rows.
9. `S2CAG feature-mode ablation 999:100`, 1 repeat: reviewer-sensitive feature table.
10. `Leiden L x r topology ablation 999:10, excluding core L=3:r=1`, 1 repeat: Pareto/frontier figure points not covered by the repeated core block.
11. `Leiden L=3 topology ablation 999:10, excluding core r=1`, 1 repeat: the remaining `L=3` frontier points.
12. `Leiden workload and closure/contraction profile`, 1 repeat: workload and closure tables.
13. `DF-Leiden workload profile`, 1 repeat: native dynamic workload points.
14. `S2CAG workload profile`, 1 repeat: GNN workload points.
15. `DSBM five-seed 100-batch core regimes`, 1 repeat: the longest block, placed last.

Expected total wall-clock is below five days under the existing cn69 timings.
The real-data blocks should finish first; the final DSBM block is intentionally
last so it can be interrupted without losing the main article-table reruns.
