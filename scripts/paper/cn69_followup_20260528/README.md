# cn69 Follow-Up DSBM Queue 2026-05-28

This follow-up queue is intentionally narrower than
`cn69_final_20260528`. The completed optional bad-locality probes already
cover the locality-admissibility discussion, so these scripts focus on the
remaining paper-facing DSBM gap: paired Leiden full-snapshot versus Local
measurements for random and hub-centered streams.

All scripts write to `results/paper_icdm/8` by default. Use a different fresh
series directory by setting `PAPER_ICDM_SERIES`, for example
`PAPER_ICDM_SERIES=9`. Before running this queue, copy the latest results into
the repository and rebuild the registry so `--skip-registry` can skip cells
that already finished in series 7.

```bash
python3 scripts/paper/collect_results_registry.py
```

Medium follow-ups:

1. `medium01_dsbm_random_5_mc1450_leiden_pair.sh`: closes the paired random
   five-batch curve at `0.05%`.
2. `medium02_dsbm_random_5_mc2900_leiden_pair.sh`: closes the random
   five-batch transition point at `0.1%`.
3. `medium03_dsbm_random_5_mc14500_leiden_pair.sh`: closes the high-update
   random five-batch point at `0.5%`.
4. `medium04_dsbm_random_5_mc29000_leiden_pair.sh`: closes the extreme random
   five-batch point at `1.0%`.
5. `medium05_dsbm_hubs_5_mc29000_leiden_pair.sh`: closes the extreme
   hub-centered five-batch point at `1.0%`.
6. `medium06_dsbm_random_10_mc290_leiden_pair.sh`: adds a low-update
   ten-batch random horizon check.
7. `medium07_dsbm_hubs_10_mc290_leiden_pair.sh`: adds a low-update
   ten-batch hub-centered horizon check.
8. `medium08_dsbm_random_hubs_10_mc1450_leiden_pair.sh`: adds the next
   ten-batch random/hub boundary point.

Heavy follow-ups:

1. `heavy01_dsbm_random_100_small_leiden_pair.sh`: random 100-batch checks at
   `0.01%` and `0.05%`; two paired cells, expected to stay near the four-hour
   envelope on cn69-class containers.
2. `heavy02_dsbm_hubs_100_small_leiden_pair.sh`: hub-centered 100-batch checks
   at `0.01%` and `0.05%`; paired with the random check above.
3. `heavy03_dsbm_random_hubs_100_mc2900_leiden_pair.sh`: random 100-batch
   check at `0.1%`.
4. `heavy04_dsbm_hubs_100_mc2900_leiden_pair.sh`: hub-centered 100-batch check
   at `0.1%`. This is split from the previous combined heavy job so the
   slowest 100-batch cells can run on separate freed containers.

Launch medium follow-ups on freed containers:

```bash
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/medium01_dsbm_random_5_mc1450_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/medium02_dsbm_random_5_mc2900_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/medium03_dsbm_random_5_mc14500_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/medium04_dsbm_random_5_mc29000_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/medium05_dsbm_hubs_5_mc29000_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/medium06_dsbm_random_10_mc290_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/medium07_dsbm_hubs_10_mc290_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/medium08_dsbm_random_hubs_10_mc1450_leiden_pair.sh'
```

Launch heavy follow-ups only after the medium queue is covered:

```bash
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/heavy01_dsbm_random_100_small_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/heavy02_dsbm_hubs_100_small_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/heavy03_dsbm_random_hubs_100_mc2900_leiden_pair.sh'
docker exec -d CT bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=8 scripts/paper/cn69_followup_20260528/heavy04_dsbm_hubs_100_mc2900_leiden_pair.sh'
```
