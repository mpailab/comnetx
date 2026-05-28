# cn69 Final Full-Picture Measurement Queue 2026-05-28

This package is generated after checking `results/registry/all_results.json`.
The current registry already contains the PubMed/Arxiv workload-profile,
closure/contraction, and random-seed blocks from `results/paper_icdm/4`, so
this queue does not repeat them.

Core paper-facing gaps:

- `tab:long-horizon` / `fig:long-horizon-curves`: newer cn69 rows already
  supply NMI for the Leiden and DF-Leiden `dyn_pubmed` `999:100` cells, but
  two LD-Leiden rows still lack final NMI. GPU0 reruns only those two LD-Leiden
  cells with `GROUND_TRUTH_METRICS=true`; it skips the block if the registry
  already contains NMI.
- `tab:contracted-workload`, `fig:workload-speedup`, `tab:closure-ablation`,
  and `tab:breakdown`: GPU1 calls the already prepared
  `scripts/paper/cn69_after3_20260527/extra27_dyn_cora_small_control.sh`.
  It skips the block if the registry already contains the required `dyn_cora`
  workload-profile cells.
- `fig:update-size`: random DSBM is missing completely, and hub-centered DSBM
  is missing except for the single existing `hubs/mc1450/100_batches`
  `leidenalg-naive` row. All DSBM core scripts pass
  `--skip-registry results/registry/all_results.json`, so already ingested
  DSBM cells are skipped.

Balanced core scripts:

- `gpu0_long_nmi_and_dsbm_random_small.sh`: missing LD-Leiden long-horizon NMI
  plus random DSBM `5_batches` and `10_batches` over all update sizes.
- `gpu1_dyn_cora_and_dsbm_hubs_small.sh`: required `dyn_cora` small-control
  profile plus hub DSBM `5_batches` and `10_batches`.
- `gpu2_dsbm_random_100_mc290.sh`: random DSBM `100_batches`, `mc=290`.
- `gpu3_dsbm_hubs_100_mc290.sh`: hub DSBM `100_batches`, `mc=290`.
- `gpu4_dsbm_random_100_mc1450.sh`: random DSBM `100_batches`, `mc=1450`.
- `gpu5_dsbm_hubs_100_mc1450.sh`: hub DSBM `100_batches`, `mc=1450`.
- `gpu6_dsbm_random_100_mid_high.sh`: random DSBM `100_batches`,
  `mc=2900,14500,29000`.
- `gpu7_dsbm_hubs_100_mid_high.sh`: hub DSBM `100_batches`,
  `mc=2900,14500,29000`.

The split uses the completed community-internal DSBM timings as a rough guide:
GPU0/GPU1 combine short mandatory completion blocks with the short DSBM
granularities, while GPU2-GPU7 split the longer `100_batches` slices.

Run from cn69 after rebuilding the registry from any newly copied results:

```bash
python3 scripts/paper/collect_results_registry.py
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu0_long_nmi_and_dsbm_random_small.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu1_dyn_cora_and_dsbm_hubs_small.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu2_dsbm_random_100_mc290.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu3_dsbm_hubs_100_mc290.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu4_dsbm_random_100_mc1450.sh'
docker exec -d dev_drobyshev bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu5_dsbm_hubs_100_mc1450.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu6_dsbm_random_100_mid_high.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu7_dsbm_hubs_100_mid_high.sh'
```

Optional bad-locality follow-ups:

The 24 `extra*.sh` scripts are deliberately not part of the core paper-facing
queue. They probe whether the method remains admissible when locality is poor
on `dyn_blogcatalog`, `dyn_wikics`, and `brain`. Run them only on containers
that finish the core queue early. Each script is a small workload-profile
probe over `999:10` and writes checkpointed JSON under `results/paper_icdm/`.

Recommended priority order for freed containers:

1. `extra09_badloc_dyn_wikics_leiden_r1.sh`: default Local Leiden on a
   bad-locality graph that is not fully saturated at radius 1.
2. `extra10_badloc_dyn_wikics_dfleiden_r1.sh`: paired DF-Leiden check for the
   same graph.
3. `extra01_badloc_dyn_blogcatalog_leiden_r1.sh`: default Local Leiden on a
   smaller but more saturated bad-locality graph.
4. `extra02_badloc_dyn_blogcatalog_dfleiden_r1.sh`: paired DF-Leiden check for
   the same graph.
5. `extra17_badloc_brain_leiden_r1.sh`: extreme dense-graph boundary check for
   Local Leiden.
6. `extra18_badloc_brain_dfleiden_r1.sh`: paired DF-Leiden dense-graph
   boundary check.
7. `extra11_badloc_dyn_wikics_leiden_r0.sh`: conservative radius-0 fallback on
   the most informative bad-locality graph.
8. `extra12_badloc_dyn_wikics_dfleiden_r0.sh`: paired DF-Leiden radius-0
   fallback.
9. `extra03_badloc_dyn_blogcatalog_leiden_r0.sh`: radius-0 fallback on the
   smaller saturated graph.
10. `extra04_badloc_dyn_blogcatalog_dfleiden_r0.sh`: paired DF-Leiden radius-0
    fallback.
11. `extra19_badloc_brain_leiden_r0.sh`: radius-0 fallback on the dense graph.
12. `extra20_badloc_brain_dfleiden_r0.sh`: paired DF-Leiden radius-0 fallback
    on the dense graph.
13. `extra15_badloc_dyn_wikics_leiden_closure_variants.sh`: closure and
    contraction diagnostic where radius 1 still leaves some signal.
14. `extra07_badloc_dyn_blogcatalog_leiden_closure_variants.sh`: closure and
    contraction diagnostic on the smaller saturated graph.
15. `extra23_badloc_brain_leiden_closure_variants.sh`: highest-risk closure and
    contraction diagnostic; run only after cheaper topology probes.
16. `extra13_badloc_dyn_wikics_leiden_r2.sh`: radius-2 blow-up check.
17. `extra14_badloc_dyn_wikics_dfleiden_r2.sh`: paired DF-Leiden radius-2
    blow-up check.
18. `extra05_badloc_dyn_blogcatalog_leiden_r2.sh`: radius-2 saturation check.
19. `extra06_badloc_dyn_blogcatalog_dfleiden_r2.sh`: paired DF-Leiden
    radius-2 saturation check.
20. `extra21_badloc_brain_leiden_r2.sh`: dense-graph radius-2 stress.
21. `extra22_badloc_brain_dfleiden_r2.sh`: paired DF-Leiden dense-graph
    radius-2 stress.
22. `extra16_badloc_dyn_wikics_s2cag_random.sh`: feature-aware probe after the
    topology envelope is known.
23. `extra08_badloc_dyn_blogcatalog_s2cag_random.sh`: feature-aware probe on
    the smaller saturated graph.
24. `extra24_badloc_brain_s2cag_random.sh`: last-resort dense feature-aware
    stress probe.

Recovery after the first core run:

`results/paper_icdm/5` contains complete `dyn_cora` workload-profile JSONs and
partial five-batch DSBM output, but the long-horizon LD-Leiden NMI cells failed
on an older `dynamic_graphs_communities` `apply()` API and the `100_batches`
DSBM jobs produced no useful measurements. Use the retry scripts below after
updating `src/baselines/dgc.py` on cn69:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/retry00_long_horizon_ldleiden_nmi.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/retry01_dsbm_random_5b_mc1450.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/retry02_dsbm_random_5b_mc2900.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/retry03_dsbm_random_5b_mc14500.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/retry04_dsbm_random_5b_mc29000.sh'
docker exec -d dev_drobyshev bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/retry05_dsbm_hubs_5b_mc290_1450.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/retry06_dsbm_hubs_5b_mc2900.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/retry07_dsbm_hubs_5b_mc14500.sh'
```

Run `retry08_dsbm_hubs_5b_mc29000.sh` on the first freed container:

```bash
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/retry08_dsbm_hubs_5b_mc29000.sh'
```

After jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```
