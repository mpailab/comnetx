# icdm-2026-0 Measurement Bundle

This directory is a self-contained supplemental measurement snapshot for
the ICDM 2026 paper iteration. It stores flattened measurement records,
including repeated runs, in semantic JSON bundles.

## Provenance

- Tag: `icdm-2026-0`
- Tagged commit: `ca2ea6fd698f78241047143ad6a033dd149dcb03`
- Total measurement entries: `4017`

## Measurement Entries By Type

- `experiment`: `3381` entries.
- `neighborhood`: `536` entries.
- `workload_profile`: `100` entries.

## Measurement Entries By Category

- `real_graph_measurements`: ICDM paper measurements on real graph streams: topology, GNN, batch-sweep, robustness, and bad-locality runs. Entries: `1430`.
- `synthetic_dsbm_measurements`: Synthetic DSBM stress measurements with controlled update streams. Entries: `246`.
- `workload_profiles`: Mechanism and workload profiling records. Entries: `100`.
- `neighborhood_measurements`: Neighborhood-size measurements used for locality and workload analysis. Entries: `536`.
- `legacy_measurements`: Earlier team, archive, imported, and traceability measurements. Entries: `1705`.

## Auxiliary Bundles

- `auxiliary/launch_manifests.json`: launch manifests for reproducibility.
- `auxiliary/launcher_errors.json`: recorded launcher failures.
- `auxiliary/registry_snapshot.json`: derived summary tables and registry snapshots.

All measurement data needed to inspect the reported runs is embedded
in this directory.
