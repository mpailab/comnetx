# cn69 ICDM Measurement Scripts

Run these scripts inside the VS Code dev container service `app` on node cn69.
They assume the dataset paths in `datasets-info/paths/cn69.json` and write
standard launcher outputs under `results/` plus logs under
`logs/paper_icdm/cn69/`.

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
- `gpu5_lago_temporal_batch_sweep.sh`: native temporal LAGO, full-snapshot
  LAGO, and ComNetX-local LAGO.
- `gpu6_dsbm_topology_stress.sh`: random, hub-centered, and community-internal
  DSBM stress streams for Leiden and DF-Leiden.
- `gpu7_dsbm_lago_stress.sh`: the same DSBM stress suite for LAGO.

After all jobs finish, rebuild the registry:

```bash
python scripts/paper/collect_results_registry.py
```
