# IEEE Access LD-Leiden: eight-container one-day campaign

This directory runs the missing native LD-Leiden comparison as eight independent,
parallel measurement shards. It does not rerun ComNetX, expand the DSBM study, or
turn the comparison into a backend leaderboard.

## Fixed allocation

| Scripts | Work per script |
|---|---|
| `01`--`05` | one fixed-input `999:10` repetition on both `dyn_pubmed` and `arxivmath` |
| `06`--`08` | one fixed-input `9:500` repetition on both `dyn_pubmed` and `arxivmath` |

The resulting evidence is exactly five short and three long repetitions for each
of the two graphs. Script `01` additionally runs the one diagnostic all-six smoke
gate; the smoke values are excluded from the aggregate. Each measured invocation
retains the two graphs together so that a container is not confounded with a
dataset.

Every shard has its own sealed campaign directory. The common validator requires
the same clean commit, source/config hashes, input hashes, physical hardware,
Python environment, LD-Leiden wheel, verified `j=1` default, and canonical
bootstrap relation. Container namespaces are intentionally different and are
checked only within their own shard. Split optimization/update/end-to-end clocks,
the priming audit, exact update counts, final modularity, and NMI are validated by
the existing registered LD-Leiden protocol.

## Required paired ComNetX campaign

`REPAIRED_CAMPAIGN_ID` is mandatory. It must identify the repaired-ComNetX
campaign whose validated short and long rows are being compared with LD-Leiden.
An arbitrary cache directory is deliberately not accepted. The launcher checks
the paired campaign manifest, commit, paths map, hardware, input manifest, and
required stage statuses before copying the canonical level-zero bootstrap into
each private LD-Leiden shard. The final validator repeats the cross-pack checks.

For the registered first-day campaign:

```bash
export REPAIRED_CAMPAIGN_ID=repaired-comnetx-cn69-20260826
```

Change that value only if the paired ComNetX measurements were intentionally run
under another campaign ID. An older ComNetX commit is accepted only when every
file in its sealed measurement-source fingerprint and every runtime dataset
metadata JSON remain byte-identical in the LD-Leiden checkout; both commit IDs
and both source-map digests are retained. Its cache must contain the six `b:999`
entries used by the smoke gate and the
`dyn_pubmed`/`arxivmath` `b:9` entries used by the long protocol. Flat and
repaired hierarchical bootstrap files are both recognized.

## Runtime controls

- Python 3.10 and an installed `dynamic_graphs_communities` wheel are required.
- The checkout must be clean and `EXPECTED_GIT_SHA` must be a full reviewed SHA.
- The first wrapper to start opens one shared window of at most 24 hours; retries
  do not reset its deadline.
- Long two-graph invocations have the preregistered one-hour hard limit.
- LD-Leiden and common numerical thread pools are fixed to one worker by the
  underlying registered runner.
- By default, the wrappers inspect each container's allowed Linux CPU topology
  and select eight evenly spaced physical cores, using one logical thread per
  core. This spreads the shards without assuming a particular EPYC SMT
  numbering. `LD_CPU_SET` may override the selected logical CPU when a container
  has a restricted cpuset. Every shard must still receive exactly one logical
  CPU; the final validator rejects shared physical cores and repeated containers.

All eight wrappers should be started promptly with the same `LD_PARALLEL_RUN_ID`.
Scripts `02`--`08` seal their environments and wait. Script `01` waits for all
preflights, validates them, runs the smoke gate, and releases the measured work.
After its own short repeat, it waits for the other seven shards and writes the
final aggregate report.

## Artifacts and monitoring

The generated tree is

```text
results/ieee-access-2026-1/raw/ldleiden/RUN_ID/
  parallel_manifest.json
  parallel_manifest_registration.json
  preflight_report.json
  smoke_gate.json
  validation_report.json
  driver-logs/01.log ... 08.log
  markers/
  shards/01-short-r01/ ... 08-long-r03/
```

Each shard retains the original launcher JSON, per-update series, stdout,
pre/post identity attestations, bootstrap provenance, and split-clock summary.
To watch progress from the host:

```bash
tail -f /home/dev/users/bokov/comnetx/results/ieee-access-2026-1/raw/ldleiden/RUN_ID/driver-logs/*.log
```

To re-run the final validation without measuring anything:

```bash
python scripts/paper/ieee_access_ldleiden_24h_parallel/validate_parallel.py \
  results/ieee-access-2026-1/raw/ldleiden/RUN_ID --stage final
```

An unfavorable LD-Leiden result is still a valid result and is never replaced.
If the three long repetitions have a time coefficient of variation above 5%,
the report flags it; any later diagnostic repeat must be appended separately and
must not replace the registered three observations.

## Eight-container launch

Run these definitions once on the Docker host after the reviewed commit is
available in the shared checkout:

```bash
REPO=/home/dev/users/bokov/comnetx
EXPECTED_GIT_SHA="$(git -C "$REPO" rev-parse HEAD)"
REPAIRED_CAMPAIGN_ID=repaired-comnetx-cn69-20260826
LD_PARALLEL_RUN_ID=ldleiden-cn69-20260827
```

Then start all eight commands promptly. The environment values are expanded by
the host shell and passed explicitly into each container.

```bash
docker exec -d dev_drobyshev3 bash -lc "cd '$REPO' && exec env EXPECTED_GIT_SHA='$EXPECTED_GIT_SHA' REPAIRED_CAMPAIGN_ID='$REPAIRED_CAMPAIGN_ID' LD_PARALLEL_RUN_ID='$LD_PARALLEL_RUN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/01_short_repeat_1.sh"
docker exec -d dev_egorov2 bash -lc "cd '$REPO' && exec env EXPECTED_GIT_SHA='$EXPECTED_GIT_SHA' REPAIRED_CAMPAIGN_ID='$REPAIRED_CAMPAIGN_ID' LD_PARALLEL_RUN_ID='$LD_PARALLEL_RUN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/02_short_repeat_2.sh"
docker exec -d dev_drobyshev2 bash -lc "cd '$REPO' && exec env EXPECTED_GIT_SHA='$EXPECTED_GIT_SHA' REPAIRED_CAMPAIGN_ID='$REPAIRED_CAMPAIGN_ID' LD_PARALLEL_RUN_ID='$LD_PARALLEL_RUN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/03_short_repeat_3.sh"
docker exec -d dev_bokov bash -lc "cd '$REPO' && exec env EXPECTED_GIT_SHA='$EXPECTED_GIT_SHA' REPAIRED_CAMPAIGN_ID='$REPAIRED_CAMPAIGN_ID' LD_PARALLEL_RUN_ID='$LD_PARALLEL_RUN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/04_short_repeat_4.sh"
docker exec -d dev_konovalov bash -lc "cd '$REPO' && exec env EXPECTED_GIT_SHA='$EXPECTED_GIT_SHA' REPAIRED_CAMPAIGN_ID='$REPAIRED_CAMPAIGN_ID' LD_PARALLEL_RUN_ID='$LD_PARALLEL_RUN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/05_short_repeat_5.sh"
docker exec -d dev_uporova bash -lc "cd '$REPO' && exec env EXPECTED_GIT_SHA='$EXPECTED_GIT_SHA' REPAIRED_CAMPAIGN_ID='$REPAIRED_CAMPAIGN_ID' LD_PARALLEL_RUN_ID='$LD_PARALLEL_RUN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/06_long_repeat_1.sh"
docker exec -d dev_egorov bash -lc "cd '$REPO' && exec env EXPECTED_GIT_SHA='$EXPECTED_GIT_SHA' REPAIRED_CAMPAIGN_ID='$REPAIRED_CAMPAIGN_ID' LD_PARALLEL_RUN_ID='$LD_PARALLEL_RUN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/07_long_repeat_2.sh"
docker exec -d dev_drobyshev bash -lc "cd '$REPO' && exec env EXPECTED_GIT_SHA='$EXPECTED_GIT_SHA' REPAIRED_CAMPAIGN_ID='$REPAIRED_CAMPAIGN_ID' LD_PARALLEL_RUN_ID='$LD_PARALLEL_RUN_ID' bash scripts/paper/ieee_access_ldleiden_24h_parallel/08_long_repeat_3.sh"
```

If any shard fails, inspect its driver log, correct the underlying problem, and
relaunch all eight commands with the same run ID. Completed attempts are reused,
failed attempts remain in the audit trail, and the original 24-hour deadline is
not reset.
