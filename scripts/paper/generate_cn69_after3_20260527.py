"""Generate the post-registry-3 cn69 measurement batch.

This package assumes that ``results/paper_icdm/3`` has already been ingested.
It avoids rerunning the completed cn69 topology, DSBM, LAGO, S2CAG dataset, and
feature-ablation sweeps. The core eight scripts target the remaining paper
gaps: workload/profile evidence, direct closure/contraction ablation, and
repeated GNN random-feature runs. The extra follow-up scripts add focused
ablation and workload checks without repeating the core batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "conf" / "paper_icdm" / "cn69_after3_20260527"
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "paper" / "cn69_after3_20260527"

FOCUSED_DATASETS = ["dyn_pubmed", "arxivmath"]
GNN_BATCHES = ["99:50", "999:50", "99:100", "999:100"]
GNN_SEEDS = [1, 2, 3, 4, 5]


def base_config() -> dict:
    return {
        "VERBOSE": 1,
        "CATCH_ERRORS": True,
        "USE_TIMESTAMP_SUFFIX": True,
        "GROUND_TRUTH_METRICS": True,
        "FORCE_UNDIRECTED": True,
        "USE_GPU": True,
    }


def launch_config(*, method: str, seed: int) -> dict:
    cfg = base_config()
    cfg.update(
        {
            "DATASETS": FOCUSED_DATASETS,
            "BATCHES": GNN_BATCHES,
            "BASELINES": [method],
            "MODES": ["smart"],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["norm"],
            },
            "FEATURE_MODES": ["random"],
            "RANDOM_FEATURE_DIM": 64,
            "RANDOM_FEATURE_SEED": seed,
            "BASELINE_ITERATIONS": [10],
        }
    )
    return cfg


def build_configs() -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for method in ["s2cag", "dmon"]:
        for seed in GNN_SEEDS:
            configs[f"{method}_smart_random_seed_{seed}.json"] = launch_config(
                method=method,
                seed=seed,
            )
    return configs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def launcher_script(*, gpu: int, title: str, configs: list[str], timeout: str) -> str:
    config_lines = "\n".join(
        f'  "conf/paper_icdm/cn69_after3_20260527/{item}"' for item in configs
    )
    tag = title.lower().replace(" ", "_").replace("/", "_")
    return f"""#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${{PARENT_HOSTNAME:-cn69}}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${{PATHS_CONFIG:-datasets-info/paths/cn69.json}}"
LOG_DIR="${{LOG_DIR:-output}}"
TIMEOUT="${{TIMEOUT:-{timeout}}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"

CONFIGS=(
{config_lines}
)

for config in "${{CONFIGS[@]}}"; do
  name="$(basename "$config" .json)"
  log="$LOG_DIR/gpu{gpu}_{tag}_${{name}}_${{STAMP}}.log"
  echo "[{title}] $(date -Is) running $config with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-container-bound}}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/launch.py "$config" --paths-config "$PATHS_CONFIG" 2>&1 | tee "$log"
  status="${{PIPESTATUS[0]}}"
  if [[ "$status" -ne 0 ]]; then
    echo "[{title}] $config exited with status $status; see $log"
    FAILED=1
  fi
done

exit "$FAILED"
"""


def profile_script(
    *,
    gpu: int | str,
    title: str,
    name: str,
    datasets: list[str],
    batches: list[str],
    methods: list[str],
    variants: list[str],
    feature_modes: list[str] | None,
    aggregation_mode: str,
    baseline_iter: int | None,
    random_seed: int | None,
    max_updates: int,
    timeout: str,
    smart_depth: int = 3,
    smart_radius: int = 1,
) -> str:
    log_prefix = f"gpu{gpu}" if isinstance(gpu, int) else str(gpu)
    feature_args = ""
    if feature_modes:
        feature_args = f" \\\n  --feature-modes {' '.join(feature_modes)}"
    iter_arg = ""
    if baseline_iter is not None:
        iter_arg = f" \\\n  --baseline-iter {baseline_iter}"
    seed_arg = ""
    if random_seed is not None:
        seed_arg = f" \\\n  --random-feature-seed {random_seed}"

    return f"""#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${{PARENT_HOSTNAME:-cn69}}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${{PATHS_CONFIG:-datasets-info/paths/cn69.json}}"
CACHE_DIR="${{CACHE_DIR:-/home/dev/communities}}"
LOG_DIR="${{LOG_DIR:-output}}"
TIMEOUT="${{TIMEOUT:-{timeout}}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

log="$LOG_DIR/{log_prefix}_{name}_${{STAMP}}.log"
echo "[{title}] $(date -Is) running workload/profile job with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-container-bound}}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py \\
  --datasets {' '.join(datasets)} \\
  --batches {' '.join(batches)} \\
  --methods {' '.join(methods)}{feature_args}{iter_arg}{seed_arg} \\
  --variants {' '.join(variants)} \\
  --smart-depth {smart_depth} \\
  --smart-radius {smart_radius} \\
  --aggregation-mode {aggregation_mode} \\
  --max-updates {max_updates} \\
  --use-gpu \\
  --force-undirected \\
  --ground-truth-metrics \\
  --catch-errors \\
  --paths-config "$PATHS_CONFIG" \\
  --cache-dir "$CACHE_DIR" \\
  --output-dir results/paper_icdm \\
  --name "{name}_${{STAMP}}" \\
  2>&1 | tee "$log"
status="${{PIPESTATUS[0]}}"
if [[ "$status" -ne 0 ]]; then
  echo "[{title}] exited with status $status; see $log"
fi
exit "$status"
"""


def dyn_cora_small_control_script() -> str:
    return r"""#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-2h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

run_profile() {
  local title="$1"
  local name="$2"
  shift 2
  local log="$LOG_DIR/extra27_${name}_${STAMP}.log"
  echo "[dyn_cora small control] $(date -Is) running ${title} with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py \
    "$@" \
    --paths-config "$PATHS_CONFIG" \
    --cache-dir "$CACHE_DIR" \
    --output-dir results/paper_icdm \
    --name "${name}_${STAMP}" \
    2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[dyn_cora small control] ${title} exited with status $status; see $log"
    FAILED=1
  fi
}

run_profile "topology full profiles" "dyn_cora_workload_topology_extra27" \
  --datasets dyn_cora \
  --batches 999:50 \
  --methods leidenalg dfleiden \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 50 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

run_profile "S2CAG full profile" "dyn_cora_workload_s2cag_extra27" \
  --datasets dyn_cora \
  --batches 999:50 \
  --methods s2cag \
  --feature-modes random \
  --baseline-iter 10 \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode norm \
  --max-updates 10 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

run_profile "Leiden closure/contraction ablation" "dyn_cora_closure_contraction_extra27" \
  --datasets dyn_cora \
  --batches 999:50 \
  --methods leidenalg \
  --variants full no_closure no_contraction \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 50 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

exit "$FAILED"
"""


def build_scripts() -> dict[str, str]:
    return {
        "gpu0_closure_contraction_pubmed.sh": profile_script(
            gpu=0,
            title="closure/contraction dyn_pubmed",
            name="after3_closure_contraction_pubmed_gpu0",
            datasets=["dyn_pubmed"],
            batches=["999:50"],
            methods=["leidenalg"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=50,
            timeout="24h",
        ),
        "gpu1_workload_memory_topology.sh": profile_script(
            gpu=1,
            title="workload/memory topology",
            name="after3_workload_memory_topology_gpu1",
            datasets=FOCUSED_DATASETS,
            batches=["999:10", "999:50", "9:500"],
            methods=["leidenalg", "dfleiden"],
            variants=["full"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=50,
            timeout="24h",
        ),
        "gpu2_workload_memory_gnn.sh": profile_script(
            gpu=2,
            title="workload/memory GNN",
            name="after3_workload_memory_gnn_gpu2",
            datasets=FOCUSED_DATASETS,
            batches=["999:10", "999:50"],
            methods=["s2cag", "dmon"],
            variants=["full"],
            feature_modes=["random"],
            aggregation_mode="norm",
            baseline_iter=10,
            random_seed=42,
            max_updates=10,
            timeout="24h",
        ),
        "gpu3_s2cag_random_seeds_1_2.sh": launcher_script(
            gpu=3,
            title="S2CAG random seeds 1-2",
            configs=[
                "s2cag_smart_random_seed_1.json",
                "s2cag_smart_random_seed_2.json",
            ],
            timeout="48h",
        ),
        "gpu4_s2cag_random_seeds_3_5.sh": launcher_script(
            gpu=4,
            title="S2CAG random seeds 3-5",
            configs=[
                "s2cag_smart_random_seed_3.json",
                "s2cag_smart_random_seed_4.json",
                "s2cag_smart_random_seed_5.json",
            ],
            timeout="72h",
        ),
        "gpu5_dmon_random_seeds_1_2.sh": launcher_script(
            gpu=5,
            title="DMoN random seeds 1-2",
            configs=[
                "dmon_smart_random_seed_1.json",
                "dmon_smart_random_seed_2.json",
            ],
            timeout="48h",
        ),
        "gpu6_dmon_random_seeds_3_5.sh": launcher_script(
            gpu=6,
            title="DMoN random seeds 3-5",
            configs=[
                "dmon_smart_random_seed_3.json",
                "dmon_smart_random_seed_4.json",
                "dmon_smart_random_seed_5.json",
            ],
            timeout="72h",
        ),
        "gpu7_closure_contraction_arxivmath.sh": profile_script(
            gpu=7,
            title="closure/contraction arxivmath",
            name="after3_closure_contraction_arxivmath_gpu7",
            datasets=["arxivmath"],
            batches=["999:50"],
            methods=["leidenalg"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=50,
            timeout="24h",
        ),
        "extra8_dfleiden_closure_contraction.sh": profile_script(
            gpu="extra8",
            title="DF-Leiden closure/contraction",
            name="after3_dfleiden_closure_contraction_extra8",
            datasets=FOCUSED_DATASETS,
            batches=["999:50"],
            methods=["dfleiden"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=50,
            timeout="24h",
        ),
        "extra9_s2cag_closure_contraction.sh": profile_script(
            gpu="extra9",
            title="S2CAG closure/contraction",
            name="after3_s2cag_closure_contraction_extra9",
            datasets=FOCUSED_DATASETS,
            batches=["999:50"],
            methods=["s2cag"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=["random"],
            aggregation_mode="norm",
            baseline_iter=10,
            random_seed=42,
            max_updates=10,
            timeout="24h",
        ),
        "extra10_leiden_radius0_pubmed.sh": profile_script(
            gpu="extra10",
            title="Leiden radius-0 dyn_pubmed workload",
            name="after3_leiden_radius0_pubmed_extra10",
            datasets=["dyn_pubmed"],
            batches=["999:50"],
            methods=["leidenalg"],
            variants=["full"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=20,
            timeout="12h",
            smart_radius=0,
        ),
        "extra11_leiden_radius2_pubmed.sh": profile_script(
            gpu="extra11",
            title="Leiden radius-2 dyn_pubmed workload",
            name="after3_leiden_radius2_pubmed_extra11",
            datasets=["dyn_pubmed"],
            batches=["999:50"],
            methods=["leidenalg"],
            variants=["full"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=20,
            timeout="12h",
            smart_radius=2,
        ),
        "extra12_s2cag_feature_modes_pubmed.sh": profile_script(
            gpu="extra12",
            title="S2CAG feature modes dyn_pubmed",
            name="after3_s2cag_feature_modes_pubmed_extra12",
            datasets=["dyn_pubmed"],
            batches=["999:50"],
            methods=["s2cag"],
            variants=["full"],
            feature_modes=["dataset", "onehot", "random"],
            aggregation_mode="norm",
            baseline_iter=10,
            random_seed=42,
            max_updates=5,
            timeout="12h",
        ),
        "extra13_leiden_radius0_arxivmath.sh": profile_script(
            gpu="extra13",
            title="Leiden radius-0 arxivmath workload",
            name="after3_leiden_radius0_arxivmath_extra13",
            datasets=["arxivmath"],
            batches=["999:50"],
            methods=["leidenalg"],
            variants=["full"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=30,
            timeout="24h",
            smart_radius=0,
        ),
        "extra14_leiden_radius2_arxivmath.sh": profile_script(
            gpu="extra14",
            title="Leiden radius-2 arxivmath workload",
            name="after3_leiden_radius2_arxivmath_extra14",
            datasets=["arxivmath"],
            batches=["999:50"],
            methods=["leidenalg"],
            variants=["full"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=30,
            timeout="24h",
            smart_radius=2,
        ),
        "extra15_dfleiden_radius0_pubmed.sh": profile_script(
            gpu="extra15",
            title="DF-Leiden radius-0 dyn_pubmed workload",
            name="after3_dfleiden_radius0_pubmed_extra15",
            datasets=["dyn_pubmed"],
            batches=["999:50"],
            methods=["dfleiden"],
            variants=["full"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=20,
            timeout="12h",
            smart_radius=0,
        ),
        "extra16_dfleiden_radius2_pubmed.sh": profile_script(
            gpu="extra16",
            title="DF-Leiden radius-2 dyn_pubmed workload",
            name="after3_dfleiden_radius2_pubmed_extra16",
            datasets=["dyn_pubmed"],
            batches=["999:50"],
            methods=["dfleiden"],
            variants=["full"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=20,
            timeout="12h",
            smart_radius=2,
        ),
        "extra17_dmon_feature_modes_pubmed.sh": profile_script(
            gpu="extra17",
            title="DMoN feature modes dyn_pubmed",
            name="after3_dmon_feature_modes_pubmed_extra17",
            datasets=["dyn_pubmed"],
            batches=["999:50"],
            methods=["dmon"],
            variants=["full"],
            feature_modes=["dataset", "onehot", "random"],
            aggregation_mode="norm",
            baseline_iter=10,
            random_seed=42,
            max_updates=5,
            timeout="12h",
        ),
        "extra18_dfleiden_radius0_arxivmath.sh": profile_script(
            gpu="extra18",
            title="DF-Leiden radius-0 arxivmath workload",
            name="after3_dfleiden_radius0_arxivmath_extra18",
            datasets=["arxivmath"],
            batches=["999:50"],
            methods=["dfleiden"],
            variants=["full"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=20,
            timeout="24h",
            smart_radius=0,
        ),
        "extra19_dfleiden_radius2_arxivmath.sh": profile_script(
            gpu="extra19",
            title="DF-Leiden radius-2 arxivmath workload",
            name="after3_dfleiden_radius2_arxivmath_extra19",
            datasets=["arxivmath"],
            batches=["999:50"],
            methods=["dfleiden"],
            variants=["full"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=20,
            timeout="24h",
            smart_radius=2,
        ),
        "extra20_dmon_closure_contraction_pubmed.sh": profile_script(
            gpu="extra20",
            title="DMoN closure/contraction dyn_pubmed",
            name="after3_dmon_closure_contraction_pubmed_extra20",
            datasets=["dyn_pubmed"],
            batches=["999:50"],
            methods=["dmon"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=["random"],
            aggregation_mode="norm",
            baseline_iter=10,
            random_seed=42,
            max_updates=5,
            timeout="12h",
        ),
        "extra21_leiden_closure_contraction_arxivmath_long.sh": profile_script(
            gpu="extra21",
            title="Leiden long closure/contraction arxivmath 99:100",
            name="after3_leiden_closure_contraction_arxivmath_99100_long_extra21",
            datasets=["arxivmath"],
            batches=["99:100"],
            methods=["leidenalg"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=120,
            timeout="9h",
        ),
        "extra22_dfleiden_closure_contraction_arxivmath_long.sh": profile_script(
            gpu="extra22",
            title="DF-Leiden long closure/contraction arxivmath 99:100",
            name="after3_dfleiden_closure_contraction_arxivmath_99100_long_extra22",
            datasets=["arxivmath"],
            batches=["99:100"],
            methods=["dfleiden"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=120,
            timeout="9h",
        ),
        "extra23_s2cag_closure_contraction_arxivmath_long.sh": profile_script(
            gpu="extra23",
            title="S2CAG long closure/contraction arxivmath 9:500",
            name="after3_s2cag_closure_contraction_arxivmath_9500_long_extra23",
            datasets=["arxivmath"],
            batches=["9:500"],
            methods=["s2cag"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=["random"],
            aggregation_mode="norm",
            baseline_iter=10,
            random_seed=42,
            max_updates=60,
            timeout="9h",
        ),
        "extra24_dfleiden_closure_variants_arxivmath_9500_long.sh": profile_script(
            gpu="extra24",
            title="DF-Leiden long closure variants arxivmath 9:500",
            name="after3_dfleiden_closure_variants_arxivmath_9500_long_extra24",
            datasets=["arxivmath"],
            batches=["9:500"],
            methods=["dfleiden"],
            variants=["no_closure", "no_contraction"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=120,
            timeout="9h",
        ),
        "extra25_dfleiden_closure_variants_arxivmath_9100_long.sh": profile_script(
            gpu="extra25",
            title="DF-Leiden long closure variants arxivmath 9:100",
            name="after3_dfleiden_closure_variants_arxivmath_9100_long_extra25",
            datasets=["arxivmath"],
            batches=["9:100"],
            methods=["dfleiden"],
            variants=["no_closure", "no_contraction"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=100,
            timeout="9h",
        ),
        "extra26_leiden_no_closure_arxivmath_9500_long.sh": profile_script(
            gpu="extra26",
            title="Leiden long no-closure arxivmath 9:500",
            name="after3_leiden_no_closure_arxivmath_9500_long_extra26",
            datasets=["arxivmath"],
            batches=["9:500"],
            methods=["leidenalg"],
            variants=["no_closure"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            random_seed=None,
            max_updates=120,
            timeout="9h",
        ),
        "extra27_dyn_cora_small_control.sh": dyn_cora_small_control_script(),
    }


README = """# cn69 After-3 Measurement Batch 2026-05-27

This package is prepared after ingesting `results/paper_icdm/3` into
`results/registry/`. It avoids the already completed topology, DSBM, LAGO,
S2CAG dataset-feature, and S2CAG feature-ablation cn69 sweeps.

The core eight GPU scripts target the remaining paper measurements:

- `gpu0_closure_contraction_pubmed.sh`: direct closure/contraction ablation for
  `dyn_pubmed`.
- `gpu1_workload_memory_topology.sh`: workload, contracted size, timing, CPU
  memory, and CUDA memory for Leiden/DF-Leiden.
- `gpu2_workload_memory_gnn.sh`: the same profiling path for S2CAG/DMoN.
- `gpu3_s2cag_random_seeds_1_2.sh`: S2CAG smart random-feature seeds 1-2.
- `gpu4_s2cag_random_seeds_3_5.sh`: S2CAG smart random-feature seeds 3-5.
- `gpu5_dmon_random_seeds_1_2.sh`: DMoN smart random-feature seeds 1-2.
- `gpu6_dmon_random_seeds_3_5.sh`: DMoN smart random-feature seeds 3-5.
- `gpu7_closure_contraction_arxivmath.sh`: direct closure/contraction ablation
  for `arxivmath`.

Optional follow-up scripts add non-duplicate ablation evidence after the core
eight are launched:

- `extra8_dfleiden_closure_contraction.sh`: direct closure/contraction ablation
  for DF-Leiden on `dyn_pubmed` and `arxivmath`.
- `extra9_s2cag_closure_contraction.sh`: direct closure/contraction ablation
  for S2CAG random features on `dyn_pubmed` and `arxivmath`.
- `extra10_leiden_radius0_pubmed.sh`: short workload profile for Leiden with
  radius 0 on `dyn_pubmed`.
- `extra11_leiden_radius2_pubmed.sh`: short workload profile for Leiden with
  radius 2 on `dyn_pubmed`.
- `extra12_s2cag_feature_modes_pubmed.sh`: short workload profile for S2CAG
  dataset, one-hot, and random features on `dyn_pubmed`.
- `extra13_leiden_radius0_arxivmath.sh`: medium workload profile for Leiden
  with radius 0 on `arxivmath`.
- `extra14_leiden_radius2_arxivmath.sh`: medium workload profile for Leiden
  with radius 2 on `arxivmath`.
- `extra15_dfleiden_radius0_pubmed.sh`: short workload profile for DF-Leiden
  with radius 0 on `dyn_pubmed`.
- `extra16_dfleiden_radius2_pubmed.sh`: short workload profile for DF-Leiden
  with radius 2 on `dyn_pubmed`.
- `extra17_dmon_feature_modes_pubmed.sh`: short workload profile for DMoN
  dataset, one-hot, and random features on `dyn_pubmed`.
- `extra18_dfleiden_radius0_arxivmath.sh`: medium workload profile for
  DF-Leiden with radius 0 on `arxivmath`.
- `extra19_dfleiden_radius2_arxivmath.sh`: medium workload profile for
  DF-Leiden with radius 2 on `arxivmath`.
- `extra20_dmon_closure_contraction_pubmed.sh`: short direct
  closure/contraction ablation for DMoN random features on `dyn_pubmed`.
- `extra21_leiden_closure_contraction_arxivmath_long.sh`: longer direct
  closure/contraction ablation for Leiden on `arxivmath`, using `99:100`.
- `extra22_dfleiden_closure_contraction_arxivmath_long.sh`: longer direct
  closure/contraction ablation for DF-Leiden on `arxivmath`, using `99:100`.
- `extra23_s2cag_closure_contraction_arxivmath_long.sh`: longer direct
  closure/contraction ablation for S2CAG random features on `arxivmath`, using
  `9:500`.
- `extra24_dfleiden_closure_variants_arxivmath_9500_long.sh`: longer
  no-closure/no-contraction DF-Leiden follow-up on `arxivmath`, using `9:500`;
  it reuses the existing full-profile baseline instead of rerunning it.
- `extra25_dfleiden_closure_variants_arxivmath_9100_long.sh`: longer
  no-closure/no-contraction DF-Leiden follow-up on `arxivmath`, using `9:100`;
  this is a smaller fallback after the `9:500` no-contraction resource limit.
- `extra26_leiden_no_closure_arxivmath_9500_long.sh`: longer no-closure Leiden
  follow-up on `arxivmath`, using `9:500`; it avoids the no-contraction branch.
- `extra27_dyn_cora_small_control.sh`: required small-graph control profile on
  `dyn_cora`. It fills the missing `dyn_cora` rows needed before adding the
  small graph to `tab:contracted-workload`, `fig:workload-speedup`,
  `tab:closure-ablation`, and `tab:breakdown`.

Run from the cn69 host with the existing GPU-bound containers:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu0_closure_contraction_pubmed.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu1_workload_memory_topology.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu2_workload_memory_gnn.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu3_s2cag_random_seeds_1_2.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu4_s2cag_random_seeds_3_5.sh'
docker exec -d dev_drobyshev bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu5_dmon_random_seeds_1_2.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu6_dmon_random_seeds_3_5.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/gpu7_closure_contraction_arxivmath.sh'
```

Run the first two follow-up scripts on any freed GPU-bound containers, for
example:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra8_dfleiden_closure_contraction.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra9_s2cag_closure_contraction.sh'
```

Run the three small follow-up scripts on the requested containers:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra10_leiden_radius0_pubmed.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra11_leiden_radius2_pubmed.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra12_s2cag_feature_modes_pubmed.sh'
```

Run the two medium follow-up scripts on the newly freed containers:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra13_leiden_radius0_arxivmath.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra14_leiden_radius2_arxivmath.sh'
```

Reserve six more follow-up scripts for any freed containers:

```bash
docker exec -d <free_container_1> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra15_dfleiden_radius0_pubmed.sh'
docker exec -d <free_container_2> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra16_dfleiden_radius2_pubmed.sh'
docker exec -d <free_container_3> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra17_dmon_feature_modes_pubmed.sh'
docker exec -d <free_container_4> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra18_dfleiden_radius0_arxivmath.sh'
docker exec -d <free_container_5> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra19_dfleiden_radius2_arxivmath.sh'
docker exec -d <free_container_6> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra20_dmon_closure_contraction_pubmed.sh'
```

Run the three longer follow-up scripts on freed containers. They use a 9h
timeout and checkpoint after every completed update/profile, so SIGTERM keeps
the latest completed partial results:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra21_leiden_closure_contraction_arxivmath_long.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra22_dfleiden_closure_contraction_arxivmath_long.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra23_s2cag_closure_contraction_arxivmath_long.sh'
```

These long scripts intentionally avoid the earlier `999:50` closure/profile
combinations: topology runs use `99:100`, and the S2CAG run uses `9:500`.

Run the next non-overlapping long follow-up on the freed `dev_konovalov`
container:

```bash
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra24_dfleiden_closure_variants_arxivmath_9500_long.sh'
```

Run the smaller non-overlapping fallback on the freed `dev_konovalov`
container if the `9:500` no-contraction branch hits a resource limit:

```bash
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra25_dfleiden_closure_variants_arxivmath_9100_long.sh'
```

Resource-limit note: `extra24` and `extra25` both showed that DF-Leiden
`no_contraction` on `arxivmath` can hit cuSPARSE insufficient resources. Keep
those results as contraction-necessity evidence and avoid scheduling more
DF-Leiden `no_contraction` arxivmath follow-ups unless the implementation or
GPU resource profile changes.

Run the next non-overlapping follow-up on the freed `dev_konovalov` container:

```bash
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra26_leiden_no_closure_arxivmath_9500_long.sh'
```

Run the required small-control profile on any freed GPU-bound container. This
is intentionally short and should be completed before inserting `dyn_cora`
into the main workload/profile tables:

```bash
docker exec -d <free_container> bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_after3_20260527/extra27_dyn_cora_small_control.sh'
```

After jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```

Check whether the after-3 jobs are still running:

```bash
for c in dev_bokov dev_uporova dev_konovalov dev_egorov dev_egorov2 dev_drobyshev dev_drobyshev2 dev_drobyshev3; do
  echo "== $c =="
  docker exec "$c" bash -lc "pgrep -af '[c]n69_after3_20260527|[a]fter3_|[p]rofile_smart_workload.py|[s]cripts/launch.py' || true"
done
```

If this only prints container headers and no PID lines, the after-3 jobs are no
longer running.

Stop the after-3 jobs without stopping the containers:

```bash
for c in dev_bokov dev_uporova dev_konovalov dev_egorov dev_egorov2 dev_drobyshev dev_drobyshev2 dev_drobyshev3; do
  docker exec "$c" bash -lc "pkill -TERM -f '[c]n69_after3_20260527|[a]fter3_' || true"
done
```

If a process ignores SIGTERM, repeat with
`pkill -KILL -f '[c]n69_after3_20260527|[a]fter3_'`.
Use SIGTERM first for checkpointed long jobs; SIGKILL cannot write a final
checkpoint, but the previous completed-update checkpoint remains on disk.

All shell logs go to `output/`. Standard launcher results go to `results/`;
profile JSON files go to `results/paper_icdm/`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-dir", default=str(CONFIG_DIR))
    parser.add_argument("--scripts-dir", default=str(SCRIPT_DIR))
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    scripts_dir = Path(args.scripts_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    for name, config in build_configs().items():
        path = configs_dir / name
        write_json(path, config)
        print(path)

    for name, content in build_scripts().items():
        path = scripts_dir / name
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | 0o755)
        print(path)

    readme = scripts_dir / "README.md"
    readme.write_text(README, encoding="utf-8")
    print(readme)


if __name__ == "__main__":
    main()
