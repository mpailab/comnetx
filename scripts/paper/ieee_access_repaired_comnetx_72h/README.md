# Repaired ComNetX: Scientific-Priority 72-Hour Queue

This package replaces, rather than silently mixes with, historical smart-mode
measurements affected by the hierarchy/namespace repair. Until a stage in this
campaign is validated, every old smart result with `L >= 2` remains
**provisional**. Full baselines are not automatically invalidated, but reuse is
allowed only under exact source, protocol, bootstrap, graph-representation,
hardware, and clock equivalence. The default queue takes the safer route and
runs one fresh paired full/smart long-horizon control.

The runner deliberately cannot create a campaign without
`--ack-production-api-settled`. This flag should be used only after the
production `Optimizer` API, hierarchy repair, and invariant tests are final.
At campaign creation, the complete Python source tree, measurement scripts,
configs, paths file, hardware record, and package environment are sealed.
Preflight content-hashes every real-stream file used by the registered configs;
Stage 6 separately seals all 30 DSBM streams and their planted labels. Every
command checks source, input, live hardware, CUDA, and installed-package
identity before and after execution. Any change makes resume or validation
fail and requires a new campaign.

## Priority and stop rules

1. **Correctness smoke (required).** Run repaired `L=3, r=1` Leiden on all six
   `999:10` streams and audit every real stream. Stop the entire queue if any
   adjacent level is non-nested before or after an update, a label is shared
   across an affected-scope boundary, an entry outside the scope changes, a
   run fails, or final metrics are non-finite.
2. **Core short evidence (required).** Run exactly five paired repetitions of
   warm-start full Leiden and repaired ComNetX on `dyn_pubmed` and
   `arxivmath`. Smoke observations are not repetitions. Unfavorable valid
   results must be retained. If neither graph has speedup above one with
   `Delta Q >= -0.05`, finish the mechanism evidence but skip optional breadth
   and reframe the empirical claim instead of tuning on these streams. This is
   a resource-allocation stop rule, not a non-inferiority margin and not a
   definition of quality preservation.
3. **Mechanism evidence (required).** First compare the instrumented `full`
   profile against the production launcher on three graphs. Any final-Q or NMI
   difference above `1e-6` is a hard stop. The profiler calls production
   `Optimizer.run(..., collect_profile=True)` rather than maintaining a second
   algorithm. Only after parity passes, collect the full method, the
   **radius-only scope** ablation (`no_closure`), and the **vertex-level base
   atoms** ablation (`no_contraction`) under `999:50`. The last ablation changes
   only level-zero atoms; higher levels remain parent quotients of the updated
   finer level. Each radius-only row is an isolated one-step counterfactual
   cloned from the production pre-update state; its result is discarded, and
   the persistent trajectory advances with the full repaired method. The gate
   requires collision-free labels, a nested hierarchy, and the same partition
   relation outside the radius-only scope. Full-row canonicalization may
   change numeric representatives outside that scope, which is recorded as a
   diagnostic and is not confused with a partition change.
4. **Long horizon (required).** Run one fresh paired `9:500` full/smart pass on
   each large graph. If time permits, add one or two smart-only repetitions;
   these are reported separately from the fresh pair.
5. **Topology and controls (optional).** Depth/radius grid comes first, then
   direction and cut metrics, then resolution. Start only with at least 38
   hours remaining. The runner terminates the current control command at the
   conservative 32-hour handoff boundary. This leaves two hours to validate,
   resume after a boundary stop, and still launch DSBM with at least 30 hours.
6. **DSBM operating regime (optional but preserved).** The registered 30 smart
   runs cover three update types, two update rates, and five seeds under
   `100_batches`. Their measured historical cost is about 26.95 GPU-hours, so
   do not start with fewer than 30 hours remaining. If the budget is
   insufficient, mark the stage `skipped_for_budget`; do not delete it and do
   not relabel the old smart rows as repaired evidence.
7. **Backend interface (last).** This stage cannot start until DSBM is either
   validated or explicitly skipped for budget. Five DF-Leiden short repeats
   are followed by one repaired smart `9:500` row. S2CAG may stop after one
   standard repaired coverage run or continue to five, and always adds the
   smart-only dataset/random/one-hot `999:100` control needed by the retained
   feature table. Historical full rows may be reused only after a separate
   exact source/protocol/bootstrap/representation/hardware/clock attestation.

These are evidence-quality gates, not outcome filters. A correct result that
weakens the paper is still a result and remains in the campaign.

## Five-RQ claim map

| Claim | Evidence that may support it |
|---|---|
| RQ1: repaired Leiden effectiveness | Stages 1, 2, and the fresh paired Stage 4 run |
| RQ2: comparator and backend generality | The separate LD-Leiden 72-hour pack is the principal comparator; Stage 7 restores DF-Leiden and S2CAG interface coverage |
| RQ3: inspected work and phase cost | Stage 3 production-backed scope, quotient-size, phase-time, memory, and restricted boundary-objective certificate profiles |
| RQ4: causal design controls | Stage 3 radius-only/vertex-atom ablations and Stage 5 depth, radius, direction, cut, and resolution controls |
| RQ5: robustness and operating boundary | Stage 4 repeatability/500-update rows, Stage 6 update-rate/type/seed grid, and the late feature-input control |

LD-Leiden evidence is intentionally owned by
`scripts/paper/ieee_access_ldleiden_72h/`; its raw results remain under the
separate `raw/ldleiden` namespace and are not mixed into this campaign.

`backend_conversion_time` in mechanism profiles is a diagnostic subcomponent
of `backend_time`. It is included once in `total_profiled_time`; the profiler
subtracts it once to expose `principal_profiled_time`. It is never added as a
second phase. The full certificate pass is retained as
`instrumented_wall_time - total_profiled_time = certificate_time`, so the
article diagnostic cannot inflate the reported production-path time.

Every Stage 3 level also records the directed-matrix quantities `W`, `W_U`,
`beta_out`, and `beta_in`, the two certificate bounds, their width, and the
realized mismatch `D` for the projected candidate. The validator independently
recomputes both bounds, checks the weight decomposition and admissible interval,
and rejects corrupted or non-finite numeric output. A zero-weight induced scope
is recorded explicitly as undefined and contributes to the reported finite-rate
coverage instead of being silently discarded. Certificate collection is
instrumentation-only and is absent from ordinary production updates.

For empirical non-vacuity, each profiled level compares the backend result with
the explicit reference `identity atom partition`: every atom of the same
quotient remains separate. The profile records the scaled local gap, restricted
full-objective gap, certificate width, tie-aware signs, and whether the strict
width threshold certifies the same full-objective sign. Stage 3 reports only
descriptive coverage and informative counts/rates; these stream levels are not
treated as independent repetitions, so no confidence interval is constructed.

The radius-only final quality value is the last registered one-step
counterfactual, not the endpoint of a multibatch radius-only trajectory. It is
used only as a paired mechanism control. Persisting `no_closure` state across
batches is intentionally unsupported by this queue.

## Commands

All commands run inside the project dev container. No measurement is launched
by merely creating this package. For the actual paired 72-hour campaign, use
the eight ordered scripts and shared absolute clock documented in
`scripts/paper/ieee_access_72h_launch/README.md`. The direct runner commands
below are low-level references; `--hours-left` alone is not a substitute for
the launcher's sealed `--deadline-epoch`.

Seal the repaired implementation. Preflight runs repository lint/unit checks,
requires a visible CUDA device and the production dynamic-backend wheel,
captures package/hardware metadata, and hashes all registered real-stream
inputs. It does not run a graph measurement:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --ack-production-api-settled --preflight
```

Run the four required stages in order, passing the actual time remaining at
each boundary:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --stage stage1_correctness_smoke --hours-left 72 --resume

python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --stage stage2_core_short --hours-left 66 --resume

python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --stage stage3_mechanism --hours-left 56 --resume

python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --stage stage4_long_core --hours-left 48 --resume
```

Optional long repeatability:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --stage stage4_long_repeatability --repetitions 2 \
  --hours-left 40 --resume
```

Controls may run only while preserving the 32-hour operational handoff to the
30-hour DSBM slot:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --stage stage5_topology_controls --hours-left 38 --resume
```

Run the preserved DSBM queue:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --stage stage6_dsbm --dsbm-root datasets-sbm \
  --hours-left 30 --resume
```

Late interface rows:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --stage stage7_dfleiden_interface --hours-left 4 --resume

python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --stage stage7_s2cag_interface --repetitions 1 \
  --hours-left 12 --resume
```

The DF-Leiden command includes five `999:10` smart repetitions plus one
smart-only `9:500` coverage run. The S2CAG command includes the requested one
to five standard repetitions plus one three-feature-mode `999:100` coverage
run.

If an optional stage cannot fit, record that decision without erasing it:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --skip-for-budget stage6_dsbm --hours-left 18 --resume
```

`--skip-for-budget` is accepted only below the stage's registered start gate,
after preflight and resolved dependencies. If Stage 2 triggers its scientific
breadth no-go, record the affected controls explicitly rather than calling the
outcome a budget decision:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id repaired-comnetx-cn69-20260825 \
  --skip-for-stage2-no-go stage5_topology_controls --resume
```

Inspect commands without executing them using `--dry-run`, or list the queue:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py --list
```

After the required core is complete:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/validate_campaign.py \
  repaired-comnetx-cn69-20260825
```

All raw outputs, stdout (with immutable digests), failed attempts, paired
bootstrap hashes and semantic level-zero checks, source fingerprints, CUDA and
package metadata, real/DSBM input manifests, hardware metadata, and stage
validation reports are stored under
`results/ieee-access-2026-1/raw/repaired-comnetx/CAMPAIGN_ID/`.
