# Reviewer follow-up queue

This queue is the single-container follow-up analogue of
`cn69_single_container_rerun_20260602`. It is intended for short reviewer-driven
additional measurements that should not run in parallel with the long article
rerun.

Launch from the server:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=15 scripts/paper/cn69_reviewer_followup_20260602/run_cut_metrics_followup.sh'
```

The runner is sequential and writes every task into
`results/paper_icdm_reviewer_followup/series_${PAPER_ICDM_SERIES}`. It creates
`.done` markers per task and can be restarted; completed tasks are skipped.

Current task:

1. `Leiden final cut metrics on key 999:10 graphs`: recomputes final Leiden
   partitions for `dyn_pubmed` and `arxivmath`, comparing full recomputation
   (`naive`) with Local ComNetX (`smart`, `L=3,r=1`). For each final partition
   it reports modularity, conductance summaries, and multiway normalized cut.

Expected runtime is dominated by full Leiden on `arxivmath` and should be close
to one clean `01_leiden_core_99910` repeat from the single-container rerun.
