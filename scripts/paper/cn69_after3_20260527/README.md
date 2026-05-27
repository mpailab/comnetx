# cn69 After-3 Measurement Batch 2026-05-27

This package is prepared after ingesting `results/paper_icdm/3` into
`results/registry/`. It avoids the already completed topology, DSBM, LAGO,
S2CAG dataset-feature, and S2CAG feature-ablation cn69 sweeps.

The eight GPU scripts target the remaining paper measurements:

- `gpu0_closure_contraction_pubmed.sh`: direct closure/contraction ablation for
  `dyn_pubmed`.
- `gpu1_workload_memory_topology.sh`: workload, contracted size, timing, CPU
  memory, and CUDA memory for Leiden/DF-Leiden.
- `gpu2_workload_memory_gnn.sh`: the same profiling path for S2CAG/DMoN.
- `gpu3_s2cag_random_seeds_1_2.sh`: S2CAG smart random-feature seeds 1-2.
- `gpu4_s2cag_random_seeds_3_5.sh`: S2CAG smart random-feature seeds 3-5.
- `gpu5_dmon_random_seeds_1_2.sh`: DMoN smart random-feature seeds 1-2.
- `gpu6_dmon_random_seeds_3_5.sh`: DMoN smart random-feature seeds 3-5.
- `gpu7_closure_contraction_arxivmath.sh`: direct closure/contraction ablation
  for `arxivmath`.

Run from the cn69 host with the existing GPU-bound containers:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu0_closure_contraction_pubmed.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu1_workload_memory_topology.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu2_workload_memory_gnn.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu3_s2cag_random_seeds_1_2.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu4_s2cag_random_seeds_3_5.sh'
docker exec -d dev_drobyshev bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu5_dmon_random_seeds_1_2.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu6_dmon_random_seeds_3_5.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu7_closure_contraction_arxivmath.sh'
```

After jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```

Check whether the eight jobs are still running:

```bash
for c in dev_bokov dev_uporova dev_konovalov dev_egorov dev_egorov2 dev_drobyshev dev_drobyshev2 dev_drobyshev3; do
  echo "== $c =="
  docker exec "$c" bash -lc "pgrep -af '[c]n69_after3_20260527|[a]fter3_|[p]rofile_smart_workload.py|[s]cripts/launch.py' || true"
done
```

If this only prints container headers and no PID lines, the after-3 jobs are no
longer running.

Stop the after-3 jobs without stopping the containers:

```bash
for c in dev_bokov dev_uporova dev_konovalov dev_egorov dev_egorov2 dev_drobyshev dev_drobyshev2 dev_drobyshev3; do
  docker exec "$c" bash -lc "pkill -TERM -f '[c]n69_after3_20260527|[a]fter3_' || true"
done
```

If a process ignores SIGTERM, repeat with
`pkill -KILL -f '[c]n69_after3_20260527|[a]fter3_'`.

All shell logs go to `output/`. Standard launcher results go to `results/`;
profile JSON files go to `results/paper_icdm/`.
