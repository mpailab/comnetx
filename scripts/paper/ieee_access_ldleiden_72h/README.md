# IEEE Access LD-Leiden 72-Hour Measurement Protocol

This package records the deliberately small LD-Leiden experiment needed for
the IEEE Access revision. It is a specialization-gap comparison, not a new
leaderboard. Native LD-Leiden is measured as an optimized, stateful C++
reference for the same community-detection family that the general Python
adapter invokes through a backend-neutral interface.

## Fixed protocol

- `smoke_999_10`: one diagnostic `999:10` run on all six real graphs. It must
  pass before measured work starts and is **never pooled with the repetitions**.
- `measured_999_10`: exactly five separate runs on both `dyn_pubmed` and
  `arxivmath`. Thus each graph has five measured observations, not six.
- `measured_9_500`: exactly three separate 500-update runs on both large
  graphs.
- Every run uses the repository's standard chronological chunks, the same
  force-undirected conversion, resolution 1, and the standard cached Leiden
  bootstrap. A campaign-local bootstrap cache makes the initial partition
  identical across repetitions. The cache must contain one canonical flat
  level-zero partition. Native priming is unmeasured, so the launcher reads
  the partition immediately afterward and stops unless its canonical relation
  is exactly the cached bootstrap relation. A wheel that cannot expose the
  post-priming partition is rejected rather than waived.
- LD-Leiden is verified at `j=1`. This is the readable default of
  `AlgorithmOptions()` in the installed wheel; the current launcher does **not**
  pass `num_jobs` explicitly. Preflight stops before any measurement if the
  property cannot be read or its default is not one. The runner also fixes
  common numerical and OpenMP thread environment variables to one.
- The principal comparison clock is `optimization_time`, which excludes the
  pending graph update inside LD-Leiden. `update_time` and `end_to_end_time`
  are retained and should be reported as diagnostics. The validator rejects
  old wheels for which `timing_split_supported` is false.

The complete pre-registration is in `protocol.json`; the three launcher inputs
are under `configs/`. Do not edit either after a campaign begins. Preflight
content-hashes every registered stream file and seals the complete
`src/**/*.py` tree, runner/validator/config sources, paths map, hardware,
container namespace, interpreter, and installed LD-Leiden Python/binary
artifacts. Every repetition rechecks all of those identities before and after
the launcher process. Resume is allowed only in the same sealed container with
unchanged data and source. A later validator may report that an archived
campaign differs from the current checkout, but the retained per-attempt and
campaign artifacts must remain internally exact.

## Measurement-server commands

Run inside one fixed measurement container. Commit the measurement code first;
a dirty tree is recorded, but a clean pinned commit is the reproducible launch
condition. For the actual 24-hour initial campaign and any append-only
extension, use the eight wrappers in
`scripts/paper/ieee_access_72h_launch/README.md`; they attach the shared window
ID and exact deadline to every attempt. The commands below are low-level API
references and are not a substitute for the bounded paper campaign. Preflight
is mandatory campaign creation and writes only identity artifacts--it does not
run a graph measurement.

Preflight the inputs, source, runtime identity, installed wheel, and readable
one-worker `AlgorithmOptions` default:

```bash
python scripts/paper/ieee_access_ldleiden_72h/run_protocol.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id ldleiden-cn69-20260826 \
  --preflight-only
```

Run the smoke gate first:

```bash
python scripts/paper/ieee_access_ldleiden_72h/run_protocol.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id ldleiden-cn69-20260826 \
  --phase smoke --resume

python scripts/paper/ieee_access_ldleiden_72h/validate_results.py \
  ldleiden-cn69-20260826 --allow-partial
```

Only after the smoke validator accepts the split clocks, run the measured
short and long phases. The same campaign id preserves the common bootstrap:

```bash
python scripts/paper/ieee_access_ldleiden_72h/run_protocol.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id ldleiden-cn69-20260826 \
  --phase short --resume

python scripts/paper/ieee_access_ldleiden_72h/run_protocol.py \
  --paths-config datasets-info/paths/cn69.json \
  --campaign-id ldleiden-cn69-20260826 \
  --phase long --resume

python scripts/paper/ieee_access_ldleiden_72h/validate_results.py \
  ldleiden-cn69-20260826
```

After preflight, an uninterrupted queue may omit `--phase` and pass `--resume`;
all phases then run in smoke, short, long order. `--resume` skips exactly one
completed attempt per repetition, retains failed attempts for audit, and never
counts them as extra repetitions.

## Generated layout

All artifacts remain below
`results/ieee-access-2026-1/raw/ldleiden/CAMPAIGN_ID/`:

```text
manifest.json
hardware.json
runtime_identity.json
real_input_manifest.json
bootstrap-cache/
smoke_999_10/repeat-01/attempt-01/
measured_999_10/repeat-01..05/attempt-01/
measured_9_500/repeat-01..03/attempt-01/
validation_report.json
```

Each attempt contains the untouched launcher JSON, stdout, pre/post identity
attestations, exact hashes, bootstrap semantics, and an analysis-ready
per-dataset summary. Bootstrap `.npz` hashes, canonical level-zero labels, and
the post-priming relation are checked across phases and repetitions. The final
validation report
contains means, sample standard deviations, and all raw aggregate values for
optimization, update, end-to-end, modularity, and NMI. Smoke observations are
explicitly marked as excluded.
