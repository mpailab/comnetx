# ICDM article-completion follow-up queue

This queue fills the remaining article measurements that are not covered by
`cn69_single_container_rerun_20260602` or
`cn69_reviewer_followup_20260602/run_cut_metrics_followup.sh`.

Launch from the server:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=16 scripts/paper/cn69_article_completion_20260602/run_article_completion_followup.sh'
```

The runner is sequential and writes every task into
`results/paper_icdm_article_completion/series_${PAPER_ICDM_SERIES}`. It creates
`.done` markers per task/repetition and can be restarted; completed tasks are
skipped.

The queue deliberately avoids overlapping the two earlier 2026-06-02 queues:

1. `Leiden dyn_cora topology L x r main grid`, 1 repeat: fills the topology
   Pareto grid for `dyn_cora` at `L in {1,2,4}` and `r in {0,1,2}`. The
   earlier single-container topology queue measured only `dyn_pubmed` and
   `arxivmath`; the already measured `dyn_cora` Full and `L=3,r=1` rows are
   not repeated here.
2. `Leiden dyn_cora topology L=3 side points`, 1 repeat: fills the remaining
   `dyn_cora` topology points `L=3,r in {0,2}` without repeating `L=3,r=1`.
3. `Leiden and DF-Leiden workload profiles 999:10`, 1 repeat: profiles the
   full ComNetX workload for `leidenalg` and `dfleiden` on `dyn_cora`,
   `dyn_pubmed`, and `arxivmath` with the same `999:10`, `L=3,r=1`,
   `aggregation=sum` protocol as the article timing rows.
4. `S2CAG dataset-feature workload profiles 999:10`, 1 repeat: profiles the
   full ComNetX workload for `s2cag` on the same three datasets with dataset
   features, `baseline_iter=10`, `L=3,r=1`, and `aggregation=sum`. This
   matches the article's S2CAG dataset-feature timing rows.

After completion, flatten these outputs into a new `icdm-2026-1` measurement
bundle file (for example
`measurements/article_completion_real_graph_measurements.json`) and regenerate
the article figures. Then all Leiden, DF-Leiden, and S2CAG article measurements
can be sourced from `results/icdm-2026-1`.
