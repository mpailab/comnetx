"""Generate cn69 measurement scripts for the ICDM paper iteration.

The generated scripts are meant for the VS Code dev container on node cn69.
They spread the paper-critical measurements over the eight available A100 GPUs
and avoid treating the short ``999:10`` setting as the main evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "conf" / "paper_icdm" / "cn69"
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "paper" / "cn69"

PRIMARY_DATASETS = [
    "dyn_cora",
    "dyn_acm",
    "dyn_citeseer",
    "patent",
    "dyn_pubmed",
    "arxivmath",
]

LONG_HORIZON_DATASETS = [
    "dyn_pubmed",
    "arxivmath",
]

FEATURE_ABLATION_DATASETS = [
    "dyn_cora",
    "dyn_pubmed",
]

REAL_BATCH_SWEEP = [
    "9:10",
    "9:50",
    "9:100",
    "99:10",
    "99:50",
    "99:100",
    "999:10",
    "999:50",
    "999:100",
]

LONG_HORIZON_BATCHES = [
    "9:200",
    "99:200",
    "999:200",
    "9:500",
    "99:500",
    "999:500",
]

GNN_BATCH_SWEEP = [
    "9:50",
    "99:50",
    "999:50",
    "9:100",
    "99:100",
    "999:100",
]


def base_config() -> dict:
    return {
        "VERBOSE": 1,
        "CATCH_ERRORS": True,
        "USE_TIMESTAMP_SUFFIX": True,
        "GROUND_TRUTH_METRICS": True,
        "FORCE_UNDIRECTED": True,
        "USE_GPU": True,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def launch_config(
    *,
    datasets: list[str],
    batches: list[str],
    baselines: list[str],
    modes: list[str],
    smart_grid: dict,
    use_gpu: bool = True,
    feature_modes: list[str] | None = None,
    random_seed: int | None = None,
    iterations: list[int] | None = None,
) -> dict:
    cfg = base_config()
    cfg.update(
        {
            "DATASETS": datasets,
            "BATCHES": batches,
            "BASELINES": baselines,
            "MODES": modes,
            "SMART_PARAMS_GRID": smart_grid,
            "USE_GPU": use_gpu,
        }
    )
    if feature_modes is not None:
        cfg["FEATURE_MODES"] = feature_modes
        cfg["RANDOM_FEATURE_DIM"] = 64
    if random_seed is not None:
        cfg["RANDOM_FEATURE_SEED"] = random_seed
    if iterations is not None:
        cfg["BASELINE_ITERATIONS"] = iterations
    return cfg


def build_configs() -> dict[str, dict]:
    configs: dict[str, dict] = {}

    configs["real_topology_batch_sweep.json"] = launch_config(
        datasets=PRIMARY_DATASETS,
        batches=REAL_BATCH_SWEEP,
        baselines=["leidenalg", "dfleiden"],
        modes=["naive", "dynamic", "smart"],
        smart_grid={
            "smart_subcoms_depth": [3],
            "smart_neighborhood_step": [1],
            "aggregation_mode": ["sum"],
        },
    )

    configs["real_topology_long_horizon.json"] = launch_config(
        datasets=LONG_HORIZON_DATASETS,
        batches=LONG_HORIZON_BATCHES,
        baselines=["leidenalg", "dfleiden"],
        modes=["naive", "dynamic", "smart"],
        smart_grid={
            "smart_subcoms_depth": [3],
            "smart_neighborhood_step": [1],
            "aggregation_mode": ["sum"],
        },
    )

    for method in ["s2cag", "dmon"]:
        configs[f"{method}_dataset_batch_sweep.json"] = launch_config(
            datasets=PRIMARY_DATASETS,
            batches=GNN_BATCH_SWEEP,
            baselines=[method],
            modes=["naive", "smart"],
            feature_modes=["dataset"],
            iterations=[10],
            smart_grid={
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["norm"],
            },
        )
        for seed in [1, 2, 3, 4, 5]:
            configs[f"{method}_random_batch_sweep_seed_{seed}.json"] = launch_config(
                datasets=PRIMARY_DATASETS,
                batches=GNN_BATCH_SWEEP,
                baselines=[method],
                modes=["naive", "smart"],
                feature_modes=["random"],
                random_seed=seed,
                iterations=[10],
                smart_grid={
                    "smart_subcoms_depth": [3],
                    "smart_neighborhood_step": [1],
                    "aggregation_mode": ["norm"],
                },
            )

    configs["feature_ablation_radius_aggregation.json"] = launch_config(
        datasets=FEATURE_ABLATION_DATASETS,
        batches=["99:50", "999:100"],
        baselines=["s2cag"],
        modes=["naive", "smart"],
        feature_modes=["dataset", "random", "onehot"],
        random_seed=42,
        iterations=[10],
        smart_grid={
            "smart_subcoms_depth": [3],
            "smart_neighborhood_step": [0, 1, 2],
            "aggregation_mode": ["norm", "sum"],
        },
    )

    configs["lago_temporal_batch_sweep.json"] = launch_config(
        datasets=PRIMARY_DATASETS,
        batches=["9:10", "99:50", "999:100"],
        baselines=["lago"],
        modes=["dynamic", "naive", "smart"],
        iterations=[1],
        use_gpu=False,
        smart_grid={
            "smart_subcoms_depth": [3],
            "smart_neighborhood_step": [1],
            "aggregation_mode": ["sum"],
        },
    )

    return configs


def launch_script(gpu: int, title: str, configs: list[str]) -> str:
    config_lines = "\n".join(f'  "conf/paper_icdm/cn69/{name}"' for name in configs)
    log_prefix = f"gpu{gpu}_{title.lower().replace(' ', '_')}"
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${{PARENT_HOSTNAME:-cn69}}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${{PATHS_CONFIG:-datasets-info/paths/cn69.json}}"
LOG_DIR="${{LOG_DIR:-output}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CONFIGS=(
{config_lines}
)

for config in "${{CONFIGS[@]}}"; do
  name="$(basename "$config" .json)"
  echo "[{title}] $(date -Is) running $config on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-container-bound}}"
  python scripts/launch.py "$config" --paths-config "$PATHS_CONFIG" 2>&1 | tee "$LOG_DIR/{log_prefix}_${{name}}_${{STAMP}}.log"
done
"""


def dsbm_script(
    *,
    gpu: int,
    title: str,
    name: str,
    methods: list[str],
    modes: list[str],
    use_gpu: bool,
) -> str:
    method_args = " ".join(methods)
    mode_args = " ".join(modes)
    gpu_arg = " --use-gpu" if use_gpu else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${{PARENT_HOSTNAME:-cn69}}"
export PYTHONUNBUFFERED=1

DSBM_ROOT="${{DSBM_ROOT:-datasets-sbm}}"
LOG_DIR="${{LOG_DIR:-output}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

if [[ ! -d "$DSBM_ROOT" ]]; then
  echo "DSBM root not found: $DSBM_ROOT"
  echo "Set DSBM_ROOT=/path/to/datasets-sbm or pass a valid datasets-sbm directory."
  exit 2
fi

echo "[{title}] $(date -Is) running DSBM stress on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-container-bound}}"
python scripts/paper/run_dsbm_stress.py \\
  --root "$DSBM_ROOT" \\
  --all-batches \\
  --methods {method_args} \\
  --modes {mode_args} \\
  --smart-depth 3 \\
  --smart-radius 1{gpu_arg} \\
  --catch-errors \\
  --output-dir results/paper_icdm \\
  --name "{name}_${{STAMP}}" \\
  2>&1 | tee "$LOG_DIR/gpu{gpu}_{name}_${{STAMP}}.log"
"""


def build_scripts() -> dict[str, str]:
    return {
        "gpu0_real_topology_batch_sweep.sh": launch_script(
            0,
            "real topology batch sweep",
            ["real_topology_batch_sweep.json"],
        ),
        "gpu1_real_topology_long_horizon.sh": launch_script(
            1,
            "real topology long horizon",
            ["real_topology_long_horizon.json"],
        ),
        "gpu2_s2cag_batch_sweep.sh": launch_script(
            2,
            "S2CAG batch and seed sweep",
            ["s2cag_dataset_batch_sweep.json"]
            + [f"s2cag_random_batch_sweep_seed_{seed}.json" for seed in [1, 2, 3, 4, 5]],
        ),
        "gpu3_dmon_batch_sweep.sh": launch_script(
            3,
            "DMoN batch and seed sweep",
            ["dmon_dataset_batch_sweep.json"]
            + [f"dmon_random_batch_sweep_seed_{seed}.json" for seed in [1, 2, 3, 4, 5]],
        ),
        "gpu4_feature_ablation_radius_aggregation.sh": launch_script(
            4,
            "feature radius and aggregation ablation",
            ["feature_ablation_radius_aggregation.json"],
        ),
        "gpu5_lago_temporal_batch_sweep.sh": launch_script(
            5,
            "LAGO temporal batch sweep",
            ["lago_temporal_batch_sweep.json"],
        ),
        "gpu6_dsbm_topology_stress.sh": dsbm_script(
            gpu=6,
            title="DSBM topology stress",
            name="dsbm_topology_stress_gpu6",
            methods=["leidenalg", "dfleiden"],
            modes=["naive", "smart", "dynamic"],
            use_gpu=True,
        ),
        "gpu7_dsbm_lago_stress.sh": dsbm_script(
            gpu=7,
            title="DSBM LAGO stress",
            name="dsbm_lago_stress_gpu7",
            methods=["lago"],
            modes=["dynamic", "naive", "smart"],
            use_gpu=False,
        ),
    }


README = """# cn69 ICDM Measurement Scripts

Run these scripts inside the cn69 Docker containers from
`/home/dev/users/bokov/comnetx`. The containers are already bound to specific
GPUs, so the scripts do not set `CUDA_VISIBLE_DEVICES` themselves. They assume
the dataset paths in `datasets-info/paths/cn69.json`, write standard launcher
outputs under `results/`, and write shell logs under `output/` by default.
Start long measurements from the host in background mode with `docker exec -d`;
do not use interactive `docker exec -it` for the paper measurement batch.

The eight scripts are intentionally complementary:

- `gpu0_real_topology_batch_sweep.sh`: real-data sensitivity over `9:*`,
  `99:*`, and `999:*` starts with 10/50/100 update batches.
- `gpu1_real_topology_long_horizon.sh`: 200/500-update long-horizon runs on
  the larger real datasets.
- `gpu2_s2cag_batch_sweep.sh`: S2CAG dataset features plus five random-feature
  seeds over the nontrivial batch sweep.
- `gpu3_dmon_batch_sweep.sh`: DMoN counterpart to the S2CAG sweep.
- `gpu4_feature_ablation_radius_aggregation.sh`: feature mode, radius, and
  aggregation ablation on representative attributed graphs.
- `gpu5_lago_temporal_batch_sweep.sh`: native temporal LAGO, full-snapshot
  LAGO, and ComNetX-local LAGO.
- `gpu6_dsbm_topology_stress.sh`: random, hub-centered, and community-internal
  DSBM stress streams for Leiden and DF-Leiden.
- `gpu7_dsbm_lago_stress.sh`: the same DSBM stress suite for LAGO.

For the two DSBM scripts, set `DSBM_ROOT` if the synthetic datasets are mounted
outside the repository checkout:

```bash
DSBM_ROOT=/path/to/datasets-sbm scripts/paper/cn69/gpu6_dsbm_topology_stress.sh
```

After all jobs finish, rebuild the registry:

```bash
python scripts/paper/collect_results_registry.py
```
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
        write_json(configs_dir / name, config)
        print(configs_dir / name)

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
