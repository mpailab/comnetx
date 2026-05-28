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

After jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```
