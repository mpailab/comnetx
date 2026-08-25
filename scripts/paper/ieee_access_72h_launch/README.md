# IEEE Access: eight container launchers

These eight scripts execute the sealed 72-hour measurement plan from inside
one fixed project container. Run them **sequentially** and in numeric order.
Every script takes a non-blocking `flock`, requires Python 3.10 and a clean git
checkout, checks the reviewed source SHA, and resumes completed attempts
without counting them twice. A failed scientific or integrity gate stops the
script and can never be relabeled automatically as a budget skip.

The shared 72-hour clock starts only after both preflights pass. Its immutable
state is stored in the repaired campaign directory, so restarting a script does
not reset the deadline. Scripts recompute the actual time remaining before
every measurement; repaired commands and LD-Leiden processes are terminated at
the deadline together with every descendant process, and no hand-entered
estimate is used. If `launch_budget.json` is lost after any measurement starts,
file 01 refuses to create a new clock; preserve the campaign and use new IDs.

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
export REPAIRED_CAMPAIGN_ID=repaired-comnetx-cn69-20260825
export LD_CAMPAIGN_ID=ldleiden-cn69-20260825
export DSBM_ROOT=datasets-sbm
export CAMPAIGN_BUDGET_HOURS=72  # may be shorter, never longer than 72
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

## Budget behavior

- Files 02--05 are the required scientific core; valid unfavorable outcomes
  are retained and never used as a stopping criterion.
- Before every LD phase, the corresponding repaired bootstrap is copied into
  the LD campaign only if the target is absent; an existing semantic mismatch
  is a hard failure. Final validation also compares both campaigns' commit,
  hardware, input hashes, and level-zero partitions.
- File 06 runs two smart-only long repetitions with at least 40 hours left, or
  one repetition in the 33--40-hour window and with otherwise stranded time in
  the 12--30-hour window. It defers repeatability only inside the 30--33-hour
  DSBM handoff window. One validated run resolves the optional evidence stage;
  the second run is a marginal precision extension. Both long
  repeatability and Stage 5 stop at the conservative 32-hour handoff boundary;
  this leaves two hours to validate and still launch DSBM at its 30-hour gate.
- File 07 runs DSBM only with at least 30 hours left; otherwise it records the
  registered budget skip.
- File 08 first resolves deferred repeatability, then the late interfaces.
  DF-Leiden requires at least 4 hours and S2CAG at least 12 hours. A Stage-2
  scientific no-go skips only optional breadth (Stage 5 and Stage 7), not
  mechanism, long-horizon, LD-Leiden, repeatability, or DSBM evidence.
- If repeatability or Stage 5 reaches the protected 32-hour boundary in the
  middle of a command, the runner records a distinct campaign-time failure and
  preserves the attempt. Re-run file 06 promptly: it resolves Stage 5 below
  38 hours while retaining the handoff margin, then continue with file 07.
- A signal interruption is recorded separately from a scientific/runtime
  failure. On resume, stale process groups are checked fail-closed before a new
  attempt starts. Fully completed commands are validated before any later
  budget-skip decision, including a completed DSBM run below its start gate.
- File 08 refuses its success message unless every optional stage is either
  validated or explicitly resolved by a registered budget/scientific rule.

Raw attempts, stdout, source/input/runtime seals, and validation reports stay
under `results/ieee-access-2026-1/raw/`. Campaign subdirectories are ignored by
git so that one preflight cannot make the next sealed campaign appear dirty;
the namespace README files remain tracked.
