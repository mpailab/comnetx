# IEEE Access LD-Leiden one-day measurements

This directory follows the launcher pattern used for the ICDM measurements:
every container runs `scripts/launch.py` directly, writes into its own result
directory, and has no dependency on another container.

## Measurements

- `01`--`05`: five repetitions on `dyn_pubmed` and `arxivmath` with `999:10`.
- `06`--`08`: three repetitions on the same graphs with `9:500`.

The two JSON configurations enable native dynamic LD-Leiden, undirected input,
ground-truth metrics, CPU execution, and split timing fields. Each repetition
keeps its own Leiden bootstrap cache under its result directory. This avoids
concurrent cache writes and retains the exact starting partition with the raw
measurement.

There is no common smoke gate, clean-check gate, manifest barrier, or
inter-container waiting. A failure in one container does not stop the other
seven. Validation and aggregation are intentionally performed after the raw
measurements have finished.

## Output

The default campaign id is `ldleiden-cn69-20260828-v3`. Each wrapper writes to:

```text
results/ieee-access-2026-1/raw/ldleiden/ldleiden-cn69-20260828-v3/
  short-r01/
    bootstrap-cache/
    ldleiden_short_999_10_now.json
    run.log
    .done or .failed
  ...
  long-r03/
```

`scripts/launch.py` checkpoints the JSON after every completed dataset. Rerunning
a failed wrapper keeps the same directory and appends to `run.log`; a wrapper
with `.done` exits immediately. Short runs have a 12-hour timeout and long runs
a 23-hour timeout. Override them with `LDLEIDEN_SHORT_TIMEOUT` and
`LDLEIDEN_LONG_TIMEOUT` if necessary.

## Launch on cn69

On the Docker host:

```bash
REPO=/home/dev/users/bokov/comnetx
LDLEIDEN_CAMPAIGN_ID=ldleiden-cn69-20260828-v3
```

Start the eight independent wrappers:

```bash
docker exec -d dev_drobyshev3 bash -lc "cd '$REPO' && exec env LDLEIDEN_CAMPAIGN_ID='$LDLEIDEN_CAMPAIGN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/01_short_repeat_1.sh"
docker exec -d dev_egorov2 bash -lc "cd '$REPO' && exec env LDLEIDEN_CAMPAIGN_ID='$LDLEIDEN_CAMPAIGN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/02_short_repeat_2.sh"
docker exec -d dev_drobyshev2 bash -lc "cd '$REPO' && exec env LDLEIDEN_CAMPAIGN_ID='$LDLEIDEN_CAMPAIGN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/03_short_repeat_3.sh"
docker exec -d dev_bokov bash -lc "cd '$REPO' && exec env LDLEIDEN_CAMPAIGN_ID='$LDLEIDEN_CAMPAIGN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/04_short_repeat_4.sh"
docker exec -d dev_konovalov bash -lc "cd '$REPO' && exec env LDLEIDEN_CAMPAIGN_ID='$LDLEIDEN_CAMPAIGN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/05_short_repeat_5.sh"
docker exec -d dev_uporova bash -lc "cd '$REPO' && exec env LDLEIDEN_CAMPAIGN_ID='$LDLEIDEN_CAMPAIGN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/06_long_repeat_1.sh"
docker exec -d dev_egorov bash -lc "cd '$REPO' && exec env LDLEIDEN_CAMPAIGN_ID='$LDLEIDEN_CAMPAIGN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/07_long_repeat_2.sh"
docker exec -d dev_drobyshev bash -lc "cd '$REPO' && exec env LDLEIDEN_CAMPAIGN_ID='$LDLEIDEN_CAMPAIGN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/08_long_repeat_3.sh"
```

Detached `docker exec -d` commands normally print nothing. Monitor all eight
logs from the host with:

```bash
tail -F "$REPO/results/ieee-access-2026-1/raw/ldleiden/$LDLEIDEN_CAMPAIGN_ID/"*/run.log
```
