# Repaired ComNetX: 24-Hour Initial Queue with Registered Extensions

This package replaces, rather than silently mixes with, historical smart-mode
measurements affected by the hierarchy/namespace repair. Until a stage in this
campaign is validated, every old smart result with `L >= 2` remains
**provisional**. Full baselines are not automatically invalidated, but reuse is
allowed only under exact source, protocol, bootstrap, graph-representation,
hardware, and clock equivalence. The default queue takes the safer route and
runs fresh pairs for both the principal long-horizon control and every retained
DSBM condition.

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
5. **Paired DSBM operating regime (protected after the core).** The registered
   design covers three update types, two update rates, and seeds 42--46 under
   `100_batches`. Every seed contains six resumable condition commands, and
   every command measures both full Leiden and repaired ComNetX from a verified
   common level-zero bootstrap: 12 fresh runs per seed. The queue advances only
   in seed order. One seed is pilot evidence, seeds 42--44 are the minimum
   publishable set, and seeds 45--46 are a precision target. Historical wall
   time is about 15.2 hours per pair-complete seed, so a 17-hour start gate is
   used. A seed interrupted after at least one condition pair resumes with a
   4-hour gate, while completed pairs are preserved. Archived full rows cannot
   be paired with new smart rows because their
   source, bootstrap, hardware, environment, and clock provenance is
   insufficient.
6. **Backend interfaces and repeatability (optional).**
   Five DF-Leiden short repeats plus one repaired smart `9:500` row come first.
   S2CAG then adds one standard repaired run and the dataset/random/one-hot
   `999:100` feature control. Two smart-only Leiden long repetitions follow.
   Before three DSBM seeds validate, File 06 protects every fresh or partial
   seed opportunity that fits in the current window and runs these only above
   that dynamic reserve, so they cannot consume the publishable-minimum path.
7. **Topology and controls (optional breadth).** After the first affordable
   three-seed DSBM minimum, run the depth/radius grid, direction and cut metrics,
   and resolution controls with a 4-hour start gate. Precision seeds wait until
   this stage and the interface/repeatability evidence are resolved. A Stage-2
   scientific no-go may skip breadth, but does not suppress DSBM or
   repeatability evidence.

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

Run the registered measurements through the eight ordered container launchers,
not by manually supplying an `--hours-left` estimate. The authoritative
first-day and extension commands are documented in
`scripts/paper/ieee_access_72h_launch/README.md`; they bind every attempt to the
current append-only window ID and exact deadline. The initial campaign ID is
`repaired-comnetx-cn69-20260826` unless explicitly overridden before preflight.

File 07 invokes Stage 6 one seed at a time with `--dsbm-seed`, preserving the
fixed order 42--46 and validating six fresh full/ComNetX condition pairs before
counting a seed. A boundary stop preserves completed condition commands, and a
later window resumes the first missing pair in the same seed. Do not use
`--skip-for-budget` for temporarily unfinished DSBM or other extension work.

The low-level queue can still be inspected without executing a measurement:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py --list
```

After any registered window, File 08 runs the repaired, LD-Leiden, and
cross-pack validators. The standalone repaired validator is:

```bash
python scripts/paper/ieee_access_repaired_comnetx_72h/validate_campaign.py \
  repaired-comnetx-cn69-20260826
```

All raw outputs, stdout (with immutable digests), failed attempts, paired
bootstrap hashes and semantic level-zero checks, source fingerprints, CUDA and
package metadata, real/DSBM input manifests, hardware metadata, and stage
validation reports are stored under
`results/ieee-access-2026-1/raw/repaired-comnetx/CAMPAIGN_ID/`.
