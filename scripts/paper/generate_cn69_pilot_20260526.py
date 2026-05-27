"""Generate the next short cn69 pilot batch for the ICDM paper.

This batch is intentionally narrower than the full cn69 suite. It follows the
10-day writing constraint in ``article/ICDM_REFACTOR_PLAN.md``: use the current
registry to probe the most valuable remaining gaps before committing to long
final runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "conf" / "paper_icdm" / "cn69_pilot_20260526"
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "paper" / "cn69_pilot_20260526"


def base_config(*, use_gpu: bool = True) -> dict:
    return {
        "VERBOSE": 1,
        "CATCH_ERRORS": True,
        "USE_TIMESTAMP_SUFFIX": True,
        "GROUND_TRUTH_METRICS": True,
        "FORCE_UNDIRECTED": True,
        "USE_GPU": use_gpu,
    }


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
    cfg = base_config(use_gpu=use_gpu)
    cfg.update(
        {
            "DATASETS": datasets,
            "BATCHES": batches,
            "BASELINES": baselines,
            "MODES": modes,
            "SMART_PARAMS_GRID": smart_grid,
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
    topology_grid = {
        "smart_subcoms_depth": [3],
        "smart_neighborhood_step": [1],
        "aggregation_mode": ["sum"],
    }
    gnn_grid = {
        "smart_subcoms_depth": [3],
        "smart_neighborhood_step": [1],
        "aggregation_mode": ["norm"],
    }

    return {
        "pilot_topology_variance_core.json": launch_config(
            datasets=["dyn_pubmed", "arxivmath"],
            batches=["999:100", "99:200"],
            baselines=["leidenalg", "dfleiden"],
            modes=["smart", "dynamic"],
            smart_grid=topology_grid,
        ),
        "pilot_s2cag_acm_smart_99_100.json": launch_config(
            datasets=["dyn_acm"],
            batches=["99:100"],
            baselines=["s2cag"],
            modes=["smart"],
            smart_grid=gnn_grid,
            feature_modes=["dataset"],
            iterations=[10],
        ),
        "pilot_s2cag_acm_999_100.json": launch_config(
            datasets=["dyn_acm"],
            batches=["999:100"],
            baselines=["s2cag"],
            modes=["naive", "smart"],
            smart_grid=gnn_grid,
            feature_modes=["dataset"],
            iterations=[10],
        ),
        "pilot_s2cag_pubmed_high_history.json": launch_config(
            datasets=["dyn_pubmed"],
            batches=["999:50", "999:100"],
            baselines=["s2cag"],
            modes=["naive", "smart"],
            smart_grid=gnn_grid,
            feature_modes=["dataset"],
            iterations=[10],
        ),
        "pilot_dmon_pubmed_high_history.json": launch_config(
            datasets=["dyn_pubmed"],
            batches=["999:50", "999:100"],
            baselines=["dmon"],
            modes=["naive", "smart"],
            smart_grid=gnn_grid,
            feature_modes=["dataset"],
            iterations=[10],
        ),
        "pilot_dmon_feature_radius_cora.json": launch_config(
            datasets=["dyn_cora"],
            batches=["99:50", "999:100"],
            baselines=["dmon"],
            modes=["naive", "smart"],
            smart_grid={
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [0, 1],
                "aggregation_mode": ["norm"],
            },
            feature_modes=["dataset", "random", "onehot"],
            random_seed=42,
            iterations=[10],
        ),
        "pilot_lago_acm_local_9_10.json": launch_config(
            datasets=["dyn_acm"],
            batches=["9:10"],
            baselines=["lago"],
            modes=["smart"],
            smart_grid=topology_grid,
            use_gpu=False,
            iterations=[1],
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def launch_script(gpu: int, title: str, configs: list[str], timeout: str = "8h") -> str:
    config_lines = "\n".join(f'  "conf/paper_icdm/cn69_pilot_20260526/{name}"' for name in configs)
    tag = f"pilot_gpu{gpu}_{title.lower().replace(' ', '_')}"
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
  log="$LOG_DIR/{tag}_${{name}}_${{STAMP}}.log"
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


def dsbm_script(
    *,
    gpu: int,
    title: str,
    name: str,
    methods: list[str],
    modes: list[str],
    regimes: list[str],
    max_changes: list[int],
    batch_suffix: str,
    use_gpu: bool,
    limit: int | None = None,
    timeout: str = "8h",
) -> str:
    method_args = " ".join(methods)
    mode_args = " ".join(modes)
    regime_args = " ".join(regimes)
    max_change_args = " ".join(str(item) for item in max_changes)
    gpu_arg = " --use-gpu" if use_gpu else ""
    limit_arg = f" \\\n  --limit {limit}" if limit is not None else ""
    return f"""#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${{PARENT_HOSTNAME:-cn69}}"
export PYTHONUNBUFFERED=1

DSBM_ROOT="${{DSBM_ROOT:-datasets-sbm}}"
LOG_DIR="${{LOG_DIR:-output}}"
TIMEOUT="${{TIMEOUT:-{timeout}}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

if [[ ! -d "$DSBM_ROOT" ]]; then
  echo "DSBM root not found: $DSBM_ROOT"
  echo "Set DSBM_ROOT=/path/to/datasets-sbm or pass a valid datasets-sbm directory."
  exit 2
fi

log="$LOG_DIR/pilot_gpu{gpu}_{name}_${{STAMP}}.log"
echo "[{title}] $(date -Is) running DSBM pilot with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-container-bound}}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/run_dsbm_stress.py \\
  --root "$DSBM_ROOT" \\
  --batch-suffix {batch_suffix} \\
  --regimes {regime_args} \\
  --max-changes {max_change_args} \\
  --methods {method_args} \\
  --modes {mode_args} \\
  --smart-depth 3 \\
  --smart-radius 1{gpu_arg} \\
  --catch-errors \\
  --output-dir results/paper_icdm \\
  --name "{name}_${{STAMP}}"{limit_arg} \\
  2>&1 | tee "$log"
status="${{PIPESTATUS[0]}}"
if [[ "$status" -ne 0 ]]; then
  echo "[{title}] DSBM pilot exited with status $status; checkpointed files, if any, are in results/paper_icdm"
fi
exit "$status"
"""


def profile_script() -> str:
    return """#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-6h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

run_profile() {
  local name="$1"
  shift
  local log="$LOG_DIR/pilot_gpu1_${name}_${STAMP}.log"
  echo "[GPU workload profile] $(date -Is) running ${name} with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py "$@" 2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[GPU workload profile] ${name} exited with status $status; see $log"
    FAILED=1
  fi
}

run_profile "topology_workload" \
  --datasets dyn_pubmed arxivmath \
  --batches 999:100 99:200 \
  --methods leidenalg dfleiden \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 20 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors \
  --paths-config "$PATHS_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output-dir results/paper_icdm \
  --name "profile_topology_workload_gpu1_${STAMP}"

run_profile "s2cag_workload" \
  --datasets dyn_cora dyn_pubmed \
  --batches 999:100 \
  --methods s2cag \
  --feature-modes dataset \
  --baseline-iter 10 \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode norm \
  --max-updates 10 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors \
  --paths-config "$PATHS_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output-dir results/paper_icdm \
  --name "profile_s2cag_workload_gpu1_${STAMP}"

exit "$FAILED"
"""


def lago_bridge_script() -> str:
    launch_part = launch_script(
        7,
        "LAGO real-data bridge",
        ["pilot_lago_acm_local_9_10.json"],
        timeout="6h",
    )
    dsbm_part = dsbm_script(
        gpu=7,
        title="LAGO DSBM feasibility",
        name="pilot_dsbm_lago_random_mc290_5b_gpu7",
        methods=["lago"],
        modes=["dynamic", "smart"],
        regimes=["random"],
        max_changes=[290],
        batch_suffix="5_batches",
        use_gpu=False,
        limit=1,
        timeout="6h",
    )
    dsbm_tail = dsbm_part.split("\n", 3)[3].replace(
        'exit "$status"',
        'if [[ "$status" -ne 0 ]]; then\n'
        '  FAILED=1\n'
        'fi\n'
        'exit "$FAILED"',
    )
    return launch_part.rsplit('exit "$FAILED"', 1)[0] + "\n" + dsbm_tail


def build_scripts() -> dict[str, str]:
    return {
        "gpu0_topology_variance_core.sh": launch_script(
            0,
            "topology variance core",
            ["pilot_topology_variance_core.json"],
            timeout="6h",
        ),
        "gpu1_workload_memory_profile.sh": profile_script(),
        "gpu2_s2cag_acm_completion.sh": launch_script(
            2,
            "S2CAG ACM completion",
            ["pilot_s2cag_acm_smart_99_100.json", "pilot_s2cag_acm_999_100.json"],
            timeout="8h",
        ),
        "gpu3_s2cag_pubmed_high_history.sh": launch_script(
            3,
            "S2CAG PubMed high-history pilot",
            ["pilot_s2cag_pubmed_high_history.json"],
            timeout="8h",
        ),
        "gpu4_dmon_pubmed_high_history.sh": launch_script(
            4,
            "DMoN PubMed high-history pilot",
            ["pilot_dmon_pubmed_high_history.json"],
            timeout="6h",
        ),
        "gpu5_dmon_feature_radius_cora.sh": launch_script(
            5,
            "DMoN feature-radius pilot",
            ["pilot_dmon_feature_radius_cora.json"],
            timeout="4h",
        ),
        "gpu6_dsbm_topology_micro.sh": dsbm_script(
            gpu=6,
            title="DSBM topology micro pilot",
            name="pilot_dsbm_topology_mc290_5b_gpu6",
            methods=["leidenalg", "dfleiden"],
            modes=["naive", "smart", "dynamic"],
            regimes=["random", "hubs", "community"],
            max_changes=[290],
            batch_suffix="5_batches",
            use_gpu=True,
            timeout="8h",
        ),
        "gpu7_lago_bridge_micro.sh": lago_bridge_script(),
    }


README = """# cn69 Pilot Batch 2026-05-26

This is the next short measurement batch selected from the current
`results/registry/` state and the 10-day ICDM writing constraint. It is a pilot
batch, not the final week-long campaign.

The batch targets five evidence gaps:

- GPU memory and workload breakdown for the method section and scalability
  tables;
- variance and long-horizon robustness for topology backends;
- high-history GNN behavior on the missing medium datasets;
- one DMoN feature/radius check to see whether S2CAG observations generalize;
- bounded DSBM checks before expanding any synthetic stress tests.

Run from the host with the usual background container pattern. The scripts
write shell logs to `output/` and result JSONs to `results/` or
`results/paper_icdm/` for DSBM and workload profiling. Each script has a
default timeout; override with `TIMEOUT=...` only when deliberately extending a
pilot.

Suggested mapping:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu0_topology_variance_core.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu1_workload_memory_profile.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu2_s2cag_acm_completion.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu3_s2cag_pubmed_high_history.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu4_dmon_pubmed_high_history.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu5_dmon_feature_radius_cora.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260526/gpu6_dsbm_topology_micro.sh'
# gpu7_lago_bridge_micro.sh is parked; do not run it for the current main paper.
```

The old `gpu7_lago_bridge_micro.sh` remains in the directory for traceability
but is no longer part of the active measurement plan. LAGO should be mentioned
in Related Work only unless a future local-temporal extension becomes a goal.

After the jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
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
