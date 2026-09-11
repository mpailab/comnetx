# IEEE Access measurement presets

This directory keeps the repaired ComNetX JSON configurations as simple
presets for the existing measurement utilities. The historical `stage*`
prefixes are names retained for provenance; they do not define a required
execution order. The eleven configurations are neither mandatory nor a
complete evaluation plan. Select the runs needed for the current claim and
record the Git revision, environment, dataset snapshot, command, and output.

Run a JSON preset directly with the standard launcher, for example:

```bash
RESULTS_DIR=results/ieee-access-2026-1/raw/repaired-comnetx/manual \
COMNETX_CACHE_DIR=tmp/communities \
python scripts/launch.py \
  scripts/paper/ieee_access_measurements/configs/stage2_paired_short_999_10.json \
  --paths-config datasets-info/paths/cn69.json
```

Profile the production smart workload and its mechanism ablations directly:

```bash
python scripts/paper/profile_smart_workload.py \
  --datasets dyn_cora dyn_pubmed arxivmath \
  --batches 999:50 \
  --methods leidenalg \
  --paths-config datasets-info/paths/cn69.json \
  --cache-dir tmp/communities \
  --output-dir results/ieee-access-2026-1/raw/repaired-comnetx/manual \
  --name mechanism \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --variants full no_closure no_contraction \
  --max-updates 50 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics
```

Run a paired DSBM slice directly (adjust the root and seeds to the available
pre-generated streams):

```bash
python scripts/paper/run_dsbm_stress.py \
  --root datasets-sbm \
  --batch-suffix 100_batches \
  --regimes random hubs community \
  --max-changes 290 1450 \
  --seeds 42 43 44 \
  --methods leidenalg \
  --modes naive smart \
  --smart-depth 3 \
  --smart-radius 1 \
  --cache-dir tmp/communities \
  --use-gpu \
  --output-dir results/ieee-access-2026-1/raw/repaired-comnetx/manual \
  --name repaired_dsbm
```

These commands write raw measurements only. They do not create a campaign,
schedule repetitions, validate admission criteria, or package article tables.
