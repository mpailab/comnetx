# cn69 Measurement Batch 2026-05-27

This batch restores the acceptance-critical direct closure/contraction
ablation and keeps placeholders only where a runnable measurement path exists.
It also matches the fixed experimental figure set in
`article/ICDM_REFACTOR_PLAN.md`.

Pilots:

- `gpu0_closure_contraction_pilot.sh`: direct `full/no_closure/no_contraction`
  Leiden pilot on `dyn_pubmed` and `arxivmath`; de-risks
  `tab:closure-ablation`.
- `gpu1_workload_memory_pilot.sh`: focused workload, contracted-size, timing,
  CPU RSS, and CUDA memory pilot for the 3 x 2 main grid; fills
  `tab:contracted-workload`, `tab:breakdown`, and `fig:workload-speedup`.
  It profiles `999:10` and a bounded prefix of `999:50` so the mechanism plot
  is not tied to a single short compatibility stream.
- `gpu2_dsbm_update_size_pilot.sh`: bounded synthetic update-size pilot across
  random, hub-centered, and community-internal regimes; de-risks
  `fig:update-size`.

Final/battle runs:

- `gpu3_topology_variance_final.sh`: fills topology focused-grid variance and
  contributes to `fig:quality-runtime-pareto`.
- `gpu4_topology_long_horizon_final.sh`: fills
  `fig:long-horizon-curves`.
- `gpu5_s2cag_focused_final.sh`: fills feature-aware focused rows and
  contributes to `fig:quality-runtime-pareto`.
- `gpu6_closure_contraction_final.sh`: final version of
  `tab:closure-ablation` after the gpu0 pilot is healthy.
- `gpu7_dsbm_update_size_final.sh`: fills `fig:update-size`.

The quality-runtime Pareto figure itself needs no separate GPU script: build it
from `results/registry/` after `gpu3` and `gpu5` are ingested.

Run pilots first if you want a conservative staged campaign. If the pilot logs
look healthy, run the corresponding final scripts. All shell logs go to
`output/`; JSON results go to `results/` or `results/paper_icdm/`.

Background container launch lines:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu0_closure_contraction_pilot.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu1_workload_memory_pilot.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu2_dsbm_update_size_pilot.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu3_topology_variance_final.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu4_topology_long_horizon_final.sh'
docker exec -d dev_drobyshev bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu5_s2cag_focused_final.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu6_closure_contraction_final.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu7_dsbm_update_size_final.sh'
```

After jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```

Monitor the eight background jobs from the host:

```bash
scripts/paper/cn69_pilot_20260527/monitor_cn69_jobs.sh
scripts/paper/cn69_pilot_20260527/monitor_cn69_jobs.sh --watch 60 --tail 3
```
