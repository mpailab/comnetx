# cn69 Pilot Batch 2026-05-26

This is the next short measurement batch selected from the current
`results/registry/` state and the 10-day ICDM writing constraint. It is a pilot
batch, not the final week-long campaign.

The batch targets five evidence gaps:

- GPU memory and workload breakdown for the method section and scalability
  tables;
- variance and long-horizon robustness for topology backends;
- high-history GNN behavior on the missing medium datasets;
- one DMoN feature/radius check to see whether S2CAG observations generalize;
- bounded DSBM checks before expanding any synthetic stress tests.

Run from the host with the usual background container pattern. The scripts
write shell logs to `output/` and result JSONs to `results/` or
`results/paper_icdm/` for DSBM and workload profiling. Each script has a
default timeout; override with `TIMEOUT=...` only when deliberately extending a
pilot.

Suggested mapping:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu0_topology_variance_core.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu1_workload_memory_profile.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu2_s2cag_acm_completion.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu3_s2cag_pubmed_high_history.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu4_dmon_pubmed_high_history.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu5_dmon_feature_radius_cora.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu6_dsbm_topology_micro.sh'
# gpu7_lago_bridge_micro.sh is parked; do not run it for the current main paper.
```

The old `gpu7_lago_bridge_micro.sh` remains in the directory for traceability
but is no longer part of the active measurement plan. LAGO should be mentioned
in Related Work only unless a future local-temporal extension becomes a goal.

After the jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```
