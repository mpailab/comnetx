# cn69 After-3 Measurement Batch 2026-05-27

This package is prepared after ingesting `results/paper_icdm/3` into
`results/registry/`. It avoids the already completed topology, DSBM, LAGO,
S2CAG dataset-feature, and S2CAG feature-ablation cn69 sweeps.

The core eight GPU scripts target the remaining paper measurements:

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

Optional follow-up scripts add non-duplicate ablation evidence after the core
eight are launched:

- `extra8_dfleiden_closure_contraction.sh`: direct closure/contraction ablation
  for DF-Leiden on `dyn_pubmed` and `arxivmath`.
- `extra9_s2cag_closure_contraction.sh`: direct closure/contraction ablation
  for S2CAG random features on `dyn_pubmed` and `arxivmath`.
- `extra10_leiden_radius0_pubmed.sh`: short workload profile for Leiden with
  radius 0 on `dyn_pubmed`.
- `extra11_leiden_radius2_pubmed.sh`: short workload profile for Leiden with
  radius 2 on `dyn_pubmed`.
- `extra12_s2cag_feature_modes_pubmed.sh`: short workload profile for S2CAG
  dataset, one-hot, and random features on `dyn_pubmed`.
- `extra13_leiden_radius0_arxivmath.sh`: medium workload profile for Leiden
  with radius 0 on `arxivmath`.
- `extra14_leiden_radius2_arxivmath.sh`: medium workload profile for Leiden
  with radius 2 on `arxivmath`.
- `extra15_dfleiden_radius0_pubmed.sh`: short workload profile for DF-Leiden
  with radius 0 on `dyn_pubmed`.
- `extra16_dfleiden_radius2_pubmed.sh`: short workload profile for DF-Leiden
  with radius 2 on `dyn_pubmed`.
- `extra17_dmon_feature_modes_pubmed.sh`: short workload profile for DMoN
  dataset, one-hot, and random features on `dyn_pubmed`.
- `extra18_dfleiden_radius0_arxivmath.sh`: medium workload profile for
  DF-Leiden with radius 0 on `arxivmath`.
- `extra19_dfleiden_radius2_arxivmath.sh`: medium workload profile for
  DF-Leiden with radius 2 on `arxivmath`.
- `extra20_dmon_closure_contraction_pubmed.sh`: short direct
  closure/contraction ablation for DMoN random features on `dyn_pubmed`.
- `extra21_leiden_closure_contraction_arxivmath_long.sh`: longer direct
  closure/contraction ablation for Leiden on `arxivmath`, using `99:100`.
- `extra22_dfleiden_closure_contraction_arxivmath_long.sh`: longer direct
  closure/contraction ablation for DF-Leiden on `arxivmath`, using `99:100`.
- `extra23_s2cag_closure_contraction_arxivmath_long.sh`: longer direct
  closure/contraction ablation for S2CAG random features on `arxivmath`, using
  `9:500`.

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

Run the first two follow-up scripts on any freed GPU-bound containers, for
example:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra8_dfleiden_closure_contraction.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra9_s2cag_closure_contraction.sh'
```

Run the three small follow-up scripts on the requested containers:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra10_leiden_radius0_pubmed.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra11_leiden_radius2_pubmed.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra12_s2cag_feature_modes_pubmed.sh'
```

Run the two medium follow-up scripts on the newly freed containers:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra13_leiden_radius0_arxivmath.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra14_leiden_radius2_arxivmath.sh'
```

Reserve six more follow-up scripts for any freed containers:

```bash
docker exec -d <free_container_1> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra15_dfleiden_radius0_pubmed.sh'
docker exec -d <free_container_2> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra16_dfleiden_radius2_pubmed.sh'
docker exec -d <free_container_3> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra17_dmon_feature_modes_pubmed.sh'
docker exec -d <free_container_4> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra18_dfleiden_radius0_arxivmath.sh'
docker exec -d <free_container_5> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra19_dfleiden_radius2_arxivmath.sh'
docker exec -d <free_container_6> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra20_dmon_closure_contraction_pubmed.sh'
```

Run the three longer follow-up scripts on freed containers. They use a 9h
timeout and checkpoint after every completed update/profile, so SIGTERM keeps
the latest completed partial results:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra21_leiden_closure_contraction_arxivmath_long.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra22_dfleiden_closure_contraction_arxivmath_long.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra23_s2cag_closure_contraction_arxivmath_long.sh'
```

These long scripts intentionally avoid the earlier `999:50` closure/profile
combinations: topology runs use `99:100`, and the S2CAG run uses `9:500`.

After jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```

Check whether the after-3 jobs are still running:

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
Use SIGTERM first for checkpointed long jobs; SIGKILL cannot write a final
checkpoint, but the previous completed-update checkpoint remains on disk.

All shell logs go to `output/`. Standard launcher results go to `results/`;
profile JSON files go to `results/paper_icdm/`.
