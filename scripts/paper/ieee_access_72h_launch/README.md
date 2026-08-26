# IEEE Access: eight container launchers

These eight scripts execute a sealed **24-hour first-day campaign** from inside
one fixed project container. If more server time becomes available, the same
campaign can be continued in append-only measurement windows up to a cumulative
72-hour cap. The directory name retains that cap; it does not mean that the
initial launch reserves three days. Run the scripts **sequentially** and in
numeric order.
Every script takes a non-blocking `flock`, requires Python 3.10 and a clean git
checkout, checks the reviewed source SHA, and resumes completed attempts
without counting them twice. A failed scientific or integrity gate stops the
script and can never be relabeled automatically as a budget skip.

The first 24-hour clock starts only after both preflights pass. Its append-only
window ledger is stored in the repaired campaign directory, so restarting a
script never resets elapsed time. Scripts recompute the actual time remaining
before every measurement; repaired commands and LD-Leiden processes are
terminated at the current window deadline together with every descendant
process, and no hand-entered runtime estimate is used. If
`launch_budget.json` is lost after any measurement starts, file 01 refuses to
create a replacement; preserve the campaign directory and investigate.

## Server setup

Check out and keep one clean commit in the already configured container. The
defaults target the `cn69` path map and the registered campaign IDs. Bind the
launch explicitly to the commit you inspected by exporting its full SHA; file
01 refuses to run without it:

```bash
cd /home/dev/users/bokov/comnetx
export EXPECTED_GIT_SHA="<full SHA from the measurement handoff>"
test "$(git rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
```

Optional overrides must be set before `01_preflight.sh` and then kept
unchanged for the whole campaign:

```bash
export PATHS_CONFIG=datasets-info/paths/cn69.json
export REPAIRED_CAMPAIGN_ID=repaired-comnetx-cn69-20260826
export LD_CAMPAIGN_ID=ldleiden-cn69-20260826
export DSBM_ROOT=datasets-sbm
export CAMPAIGN_BUDGET_HOURS=24  # the initial window is fixed at 24 hours
```

## Launch order

```bash
bash scripts/paper/ieee_access_72h_launch/01_preflight.sh
bash scripts/paper/ieee_access_72h_launch/02_smoke_gates.sh
bash scripts/paper/ieee_access_72h_launch/03_short_core.sh
bash scripts/paper/ieee_access_72h_launch/04_mechanism.sh
bash scripts/paper/ieee_access_72h_launch/05_long_core.sh
bash scripts/paper/ieee_access_72h_launch/06_repeatability_and_controls.sh
bash scripts/paper/ieee_access_72h_launch/07_dsbm.sh
bash scripts/paper/ieee_access_72h_launch/08_interfaces_and_final_validation.sh
```

Do not run two files in parallel, in two containers, or after changing source,
configuration, input data, hardware, the Python interpreter, or an installed
backend wheel. The campaign runners verify these identities again before and
after every recorded attempt.

## First-day scientific order

- Files 02--05 are the required scientific core; valid unfavorable outcomes
  are retained and never used as a stopping criterion. The repaired pack and
  all three registered LD-Leiden phases keep their original repetition counts.
- Before every LD phase, the corresponding repaired bootstrap is copied into
  the LD campaign only if the target is absent; an existing semantic mismatch
  is a hard failure. Final validation also compares both campaigns' commit,
  hardware, input hashes, and level-zero partitions.
- File 06 spends any post-core time on the repaired backend interfaces first:
  DF-Leiden has a 1-hour start gate, followed by one S2CAG repetition plus its
  feature-input control with a 2-hour gate. Until three paired DSBM seeds are
  complete, File 06 first protects every 17-hour fresh-seed opportunity that
  fits in the active window, plus 4-hour condition-pair increments that can
  advance an interrupted seed, and runs optional work only above that dynamic
  reserve. It then gives the two
  registered smart-only 9:500 repeats a 2-hour allowance above the same reserve.
  A command that completed just before the boundary is finalized; other
  unfinished work remains pending for an extension window.
- File 07 uses every available 17-hour slot to advance the paired DSBM prefix
  to seeds 42--44 before breadth controls. Each seed contains six resumable
  condition commands covering all three update types at both update rates; each
  command measures a fresh full/ComNetX pair, for 12 runs per completed seed.
  Only a seed with all six validated pairs counts. After the three-seed minimum,
  the topology, direction, cut, and resolution controls use a 4-hour gate;
  precision seeds 45--46 wait until interface, repeatability, and Stage 5
  evidence are resolved.
- A fresh seed starts only with 17 hours left. If a deadline stops it after one
  or more complete condition pairs, a later window may resume that same seed
  with a 4-hour gate; at most the currently running condition pair is repeated.
- Historical DSBM full rows are not denominators for these new smart rows: the
  archive does not attest the exact source, bootstrap, hardware, environment,
  and clock identity required for a timing pair. Stage 6 therefore computes
  speedup, $\Delta Q$, and $\Delta$NMI only from contemporaneous validated pairs.
- One DSBM seed is only pilot evidence. Do not use it as the final robustness
  result. Three complete fixed seeds are the minimum publishable evidence and
  all five are the precision target. Historical timing indicates about 15.2
  wall-clock hours per paired seed, so the initial day can normally produce at
  most one pilot and the 72-hour aggregate cap does not guarantee all five.
  Missing seeds remain pending and are never disguised as a budget skip.
- A Stage-2 scientific no-go skips only the optional breadth interfaces and
  Stage 5 controls. It does not suppress the required core, LD-Leiden,
  mechanism, long-horizon, repeatability, or DSBM evidence.
- File 08 runs validators and records an end-of-window checkpoint. It succeeds
  when the complete core and all three LD-Leiden phases validate, even if an
  optional stage remains pending. Its success message is not a measurement
  freeze and explicitly reports whether DSBM still needs more seeds.
- A signal interruption is recorded separately from a scientific/runtime
  failure. On resume, stale process groups are checked fail-closed before a new
  attempt starts. Fully completed commands and seed blocks are validated before
  later work begins.

## Continuing after the first day

An extension is an additional registered measurement window, not a reset of
the first-day clock. Wait until the active window has expired, keep the exact
same checkout, campaign IDs, path map, inputs, container environment, hardware,
and backend wheels, then open the next window through file 01:

```bash
export CAMPAIGN_EXTENSION_HOURS=30
bash scripts/paper/ieee_access_72h_launch/01_preflight.sh
# The 06--08 shortcut below is valid only after File 05 and all LD phases validate.
bash scripts/paper/ieee_access_72h_launch/06_repeatability_and_controls.sh
bash scripts/paper/ieee_access_72h_launch/07_dsbm.sh
bash scripts/paper/ieee_access_72h_launch/08_interfaces_and_final_validation.sh
unset CAMPAIGN_EXTENSION_HOURS
```

If the previous window ended inside the required core or an LD-Leiden phase,
re-run sequentially from the earliest unfinished file among 02--08 instead of
using the 06--08 shortcut. Completed attempts are validated and skipped.

The ledger assigns the next immutable identifier (for example `window-002`),
rejects overlapping windows, and rejects grants that would exceed 72 aggregate
hours. File 06 resumes work without crossing the protected paired-DSBM reserve.
File 07 resumes completed condition pairs within the next fixed seed, reaches
the three-seed minimum before Stage 5, and treats seeds 45--46 as precision
extensions rather than displacing other registered evidence. File 08 can be run
after every window to regenerate the
reproducibility reports. If only DSBM remains, files 06 and 07 are safe to
re-run: file 06 reports its already validated stages and file 07 continues the
seed queue without duplicating completed blocks.

Raw attempts, stdout, source/input/runtime seals, and validation reports stay
under `results/ieee-access-2026-1/raw/`. Campaign subdirectories are ignored by
git so that one preflight cannot make the next sealed campaign appear dirty;
the namespace README files remain tracked.
