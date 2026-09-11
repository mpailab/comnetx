# Repaired ComNetX Raw Reruns

This directory holds new measurements of the implementation introduced in
`1061555`. Use the existing project launch tools and the configurations
documented in `scripts/paper/ieee_access_measurements/README.md`.

Keep each run's source revision, configuration, input identity, environment,
shared bootstrap, log, and result together. Preserve failed and unfavorable
runs. Historical ICDM smart-mode measurements describe the earlier update
procedure and must not be relabeled as runs of the changed implementation.
The namespace collision can also affect depth one; depth alone is not a
certificate that an old result is unaffected.

The obsolete 24/72-hour scheduler and its coupled LD-Leiden requirements were
removed. No new server campaign has been launched by the code audit.
