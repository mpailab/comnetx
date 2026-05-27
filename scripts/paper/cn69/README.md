# cn69 ICDM Measurement Scripts

Run these scripts inside the cn69 Docker containers from
`/home/dev/users/bokov/comnetx`. The containers are already bound to specific
GPUs, so the scripts do not set `CUDA_VISIBLE_DEVICES` themselves. They assume
the dataset paths in `datasets-info/paths/cn69.json`, write standard launcher
outputs under `results/`, and write shell logs under `output/` by default.
Start long measurements from the host in background mode with `docker exec -d`;
do not use interactive `docker exec -it` for the paper measurement batch.

With the current 10-day paper sprint, treat this package as a maximal candidate
suite, not as an immediate all-at-once schedule. First ingest the latest
available result JSONs, rebuild `results/registry/`, run short pilots that
finish in hours or within one day, and only then schedule final multi-day jobs
for the strongest remaining evidence gaps.

The eight scripts are intentionally complementary:

- `gpu0_real_topology_batch_sweep.sh`: real-data sensitivity over `9:*`,
  `99:*`, and `999:*` starts with 10/50/100 update batches.
- `gpu1_real_topology_long_horizon.sh`: 200/500-update long-horizon runs on
  the larger real datasets.
- `gpu2_s2cag_batch_sweep.sh`: S2CAG dataset features plus five random-feature
  seeds over the nontrivial batch sweep.
- `gpu3_dmon_batch_sweep.sh`: DMoN counterpart to the S2CAG sweep.
- `gpu4_feature_ablation_radius_aggregation.sh`: feature mode, radius, and
  aggregation ablation on representative attributed graphs.
- `gpu5_lago_temporal_batch_sweep.sh`: parked optional LAGO script from an
  earlier plan; do not run it for the current main paper evidence chain.
- `gpu6_dsbm_topology_stress.sh`: random, hub-centered, and community-internal
  DSBM stress streams for Leiden and DF-Leiden.
- `gpu7_dsbm_lago_stress.sh`: parked optional LAGO stress script from an
  earlier plan; do not run it for the current main paper evidence chain.

For the two DSBM scripts, set `DSBM_ROOT` if the synthetic datasets are mounted
outside the repository checkout:

```bash
DSBM_ROOT=/path/to/datasets-sbm scripts/paper/cn69/gpu6_dsbm_topology_stress.sh
```

The DSBM runner checkpoints before the first algorithm call and after every
completed or failed method/dataset attempt. It writes `manifest_<run>.json`
alongside the result JSON so interrupted jobs preserve completed measurements
and expose the current attempt.

After all jobs finish, rebuild the registry:

```bash
python scripts/paper/collect_results_registry.py
```
