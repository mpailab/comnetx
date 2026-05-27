"""Generate the 2026-05-27 cn69 measurement batch for the ICDM paper.

The batch keeps real pilots only where runtime is uncertain and uses the
remaining GPU slots for higher-value final measurements. All scripts are meant
to be launched inside the existing GPU-bound Docker containers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "conf" / "paper_icdm" / "cn69_pilot_20260527"
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "paper" / "cn69_pilot_20260527"


def base_config() -> dict:
    return {
        "VERBOSE": 1,
        "CATCH_ERRORS": True,
        "USE_TIMESTAMP_SUFFIX": True,
        "GROUND_TRUTH_METRICS": True,
        "FORCE_UNDIRECTED": True,
        "USE_GPU": True,
    }


def launch_config(
    *,
    datasets: list[str],
    batches: list[str],
    baselines: list[str],
    modes: list[str],
    smart_grid: dict,
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
    s2cag_grid = {
        "smart_subcoms_depth": [3],
        "smart_neighborhood_step": [1],
        "aggregation_mode": ["norm"],
    }
    return {
        "topology_variance_final.json": launch_config(
            datasets=["dyn_pubmed", "arxivmath"],
            batches=["999:10", "999:50", "999:100", "99:100"],
            baselines=["leidenalg", "dfleiden"],
            modes=["naive", "smart", "dynamic"],
            smart_grid=topology_grid,
        ),
        "topology_long_horizon_final.json": launch_config(
            datasets=["dyn_pubmed", "arxivmath"],
            batches=["999:200", "99:200", "999:500", "99:500"],
            baselines=["leidenalg", "dfleiden"],
            modes=["naive", "smart", "dynamic"],
            smart_grid=topology_grid,
        ),
        "s2cag_naive_dataset_final.json": launch_config(
            datasets=["dyn_pubmed", "arxivmath"],
            batches=["999:10", "999:50"],
            baselines=["s2cag"],
            modes=["naive"],
            smart_grid=s2cag_grid,
            feature_modes=["dataset"],
            iterations=[10],
        ),
        "s2cag_smart_random_final.json": launch_config(
            datasets=["dyn_pubmed", "arxivmath"],
            batches=["999:10", "999:50"],
            baselines=["s2cag"],
            modes=["smart"],
            smart_grid=s2cag_grid,
            feature_modes=["random"],
            random_seed=42,
            iterations=[10],
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def launcher_script(gpu: int, title: str, configs: list[str], timeout: str) -> str:
    config_lines = "\n".join(f'  "conf/paper_icdm/cn69_pilot_20260527/{item}"' for item in configs)
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
    gpu: int,
    title: str,
    name: str,
    datasets: list[str],
    batches: list[str],
    methods: list[str],
    variants: list[str],
    feature_modes: list[str] | None,
    aggregation_mode: str,
    baseline_iter: int | None,
    max_updates: int,
    timeout: str,
) -> str:
    feature_args = f" \\\n+  --feature-modes {' '.join(feature_modes)}" if feature_modes else ""
    iter_arg = f" \\\n+  --baseline-iter {baseline_iter}" if baseline_iter is not None else ""
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

log="$LOG_DIR/gpu{gpu}_{name}_${{STAMP}}.log"
echo "[{title}] $(date -Is) running workload/profile job with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-container-bound}}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py \\
  --datasets {' '.join(datasets)} \\
  --batches {' '.join(batches)} \\
  --methods {' '.join(methods)}{feature_args}{iter_arg} \\
  --variants {' '.join(variants)} \\
  --smart-depth 3 \\
  --smart-radius 1 \\
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


def profile_combo_script() -> str:
    return """#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-24h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

run_profile() {
  local name="$1"
  shift
  local log="$LOG_DIR/gpu1_${name}_${STAMP}.log"
  echo "[workload/memory pilot] $(date -Is) running ${name} with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py "$@" 2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[workload/memory pilot] ${name} exited with status $status; see $log"
    FAILED=1
  fi
}

run_profile "topology_workload_memory" \
  --datasets dyn_pubmed arxivmath \
  --batches 999:10 999:50 \
  --methods leidenalg dfleiden \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 50 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors \
  --paths-config "$PATHS_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output-dir results/paper_icdm \
  --name "profile_topology_workload_memory_gpu1_${STAMP}"

run_profile "s2cag_workload_memory" \
  --datasets dyn_pubmed arxivmath \
  --batches 999:10 999:50 \
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
  --catch-errors \
  --paths-config "$PATHS_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output-dir results/paper_icdm \
  --name "profile_s2cag_workload_memory_gpu1_${STAMP}"

exit "$FAILED"
"""


def dsbm_script(
    *,
    gpu: int,
    title: str,
    name: str,
    batch_suffix: str,
    max_changes: list[int],
    timeout: str,
    limit: int | None = None,
) -> str:
    limit_arg = f" \\\n+  --limit {limit}" if limit is not None else ""
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
  exit 2
fi

log="$LOG_DIR/gpu{gpu}_{name}_${{STAMP}}.log"
echo "[{title}] $(date -Is) running DSBM job with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-container-bound}}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/run_dsbm_stress.py \\
  --root "$DSBM_ROOT" \\
  --batch-suffix {batch_suffix} \\
  --regimes random hubs community \\
  --max-changes {' '.join(str(item) for item in max_changes)} \\
  --methods leidenalg dfleiden \\
  --modes naive smart dynamic \\
  --smart-depth 3 \\
  --smart-radius 1 \\
  --use-gpu \\
  --catch-errors \\
  --output-dir results/paper_icdm \\
  --name "{name}_${{STAMP}}"{limit_arg} \\
  2>&1 | tee "$log"
status="${{PIPESTATUS[0]}}"
if [[ "$status" -ne 0 ]]; then
  echo "[{title}] exited with status $status; checkpointed files, if any, are in results/paper_icdm"
fi
exit "$status"
"""


def monitor_script() -> str:
    return r"""#!/usr/bin/env bash
set -uo pipefail

REMOTE_WORKDIR="${REMOTE_WORKDIR:-/home/dev/users/bokov/comnetx}"
WATCH_INTERVAL=0
TAIL_LINES=0
PROCESS_PATTERN='[s]cripts/launch.py|[p]rofile_smart_workload.py|[r]un_dsbm_stress.py'

CONTAINERS=(
  dev_bokov
  dev_uporova
  dev_konovalov
  dev_egorov
  dev_egorov2
  dev_drobyshev
  dev_drobyshev2
  dev_drobyshev3
)

GPUS=(0 1 2 3 4 5 6 7)

SCRIPTS=(
  gpu0_closure_contraction_pilot.sh
  gpu1_workload_memory_pilot.sh
  gpu2_dsbm_update_size_pilot.sh
  gpu3_topology_variance_final.sh
  gpu4_topology_long_horizon_final.sh
  gpu5_s2cag_focused_final.sh
  gpu6_closure_contraction_final.sh
  gpu7_dsbm_update_size_final.sh
)

usage() {
  cat <<'EOF'
Usage: scripts/paper/cn69_pilot_20260527/monitor_cn69_jobs.sh [options]

Options:
  --watch SECONDS   Repeat the status check every SECONDS.
  --tail LINES      Show the last LINES lines of each latest GPU log.
  -h, --help        Show this help.

Environment:
  REMOTE_WORKDIR    Project directory inside each container.
                   Default: /home/dev/users/bokov/comnetx
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)
      WATCH_INTERVAL="${2:-}"
      shift 2
      ;;
    --tail)
      TAIL_LINES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$WATCH_INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "--watch must be a non-negative integer" >&2
  exit 2
fi

if ! [[ "$TAIL_LINES" =~ ^[0-9]+$ ]]; then
  echo "--tail must be a non-negative integer" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command is not available" >&2
  exit 127
fi

container_state() {
  local container="$1"
  docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true
}

running_processes() {
  local container="$1"
  docker exec "$container" bash -lc "pgrep -af '$PROCESS_PATTERN' || true" 2>/dev/null || true
}

remote_log_info() {
  local container="$1"
  local gpu="$2"
  docker exec "$container" bash -lc "
    cd '$REMOTE_WORKDIR' 2>/dev/null || { echo 'NO_WORKDIR'; exit 0; }
    latest_log=\$(ls -t output/gpu${gpu}_*.log 2>/dev/null | head -1 || true)
    if [[ -z \"\$latest_log\" ]]; then
      echo 'LOG: none'
      echo 'LOG_MTIME: n/a'
      echo 'LAST: n/a'
      echo 'ERROR_HINT: no'
    else
      echo \"LOG: \$latest_log\"
      echo \"LOG_MTIME: \$(stat -c '%y' \"\$latest_log\" 2>/dev/null | cut -d'.' -f1)\"
      echo \"LAST: \$(tail -n 1 \"\$latest_log\" 2>/dev/null | tr '\t' ' ')\"
      if tail -n 100 \"\$latest_log\" 2>/dev/null | grep -Eiq 'exited with status|Traceback|ModuleNotFoundError|RuntimeError|CUDA.*error|Killed|timed out|No such file|not found|Error on '; then
        echo 'ERROR_HINT: yes'
      else
        echo 'ERROR_HINT: no'
      fi
      if [[ '$TAIL_LINES' -gt 0 ]]; then
        echo 'TAIL_BEGIN'
        tail -n '$TAIL_LINES' \"\$latest_log\" 2>/dev/null
        echo 'TAIL_END'
      fi
    fi
    latest_json=\$(find results results/paper_icdm -maxdepth 2 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -3 | cut -d' ' -f2- || true)
    if [[ -n \"\$latest_json\" ]]; then
      echo 'RECENT_JSON:'
      echo \"\$latest_json\" | sed 's/^/  /'
    else
      echo 'RECENT_JSON: none'
    fi
  " 2>&1 || true
}

field_value() {
  local name="$1"
  awk -F': ' -v key="$name" '$1 == key {print substr($0, length(key) + 3); exit}'
}

run_once() {
  echo "cn69 paper job status at $(date -Is)"
  printf '%-16s %-5s %-36s %-14s %s\n' "container" "gpu" "script" "status" "latest_log"
  printf '%-16s %-5s %-36s %-14s %s\n' "---------" "---" "------" "------" "----------"

  local idx
  for idx in "${!CONTAINERS[@]}"; do
    local container="${CONTAINERS[$idx]}"
    local gpu="${GPUS[$idx]}"
    local script_name="${SCRIPTS[$idx]}"
    local state
    state="$(container_state "$container")"

    if [[ "$state" != "true" ]]; then
      printf '%-16s %-5s %-36s %-14s %s\n' "$container" "gpu$gpu" "$script_name" "CONTAINER_DOWN" "n/a"
      continue
    fi

    local processes
    processes="$(running_processes "$container" "$script_name")"

    local info
    info="$(remote_log_info "$container" "$gpu")"
    local latest_log
    latest_log="$(printf '%s\n' "$info" | field_value LOG)"
    local log_mtime
    log_mtime="$(printf '%s\n' "$info" | field_value LOG_MTIME)"
    local last_line
    last_line="$(printf '%s\n' "$info" | field_value LAST)"
    local error_hint
    error_hint="$(printf '%s\n' "$info" | field_value ERROR_HINT)"

    local status
    if [[ -n "$processes" ]]; then
      status="RUNNING"
    elif [[ "$latest_log" == "none" || -z "$latest_log" ]]; then
      status="NOT_STARTED"
    elif [[ "$error_hint" == "yes" ]]; then
      status="CHECK_LOG"
    else
      status="DONE"
    fi

    printf '%-16s %-5s %-36s %-14s %s\n' "$container" "gpu$gpu" "$script_name" "$status" "$latest_log"
    echo "  log_mtime: ${log_mtime:-n/a}"
    echo "  last_line: ${last_line:-n/a}"
    if [[ "$status" == "RUNNING" ]]; then
      echo "$processes" | sed 's/^/  process: /'
    fi
    printf '%s\n' "$info" | awk '
      /^RECENT_JSON:/ {show=1; print "  recent_json:"; next}
      /^TAIL_BEGIN$/ {tail=1; print "  tail:"; next}
      /^TAIL_END$/ {tail=0; next}
      show && /^  / {print "  " $0; next}
      show && !/^  / {show=0}
      tail {print "    " $0}
    '
  done
}

while true; do
  run_once
  if [[ "$WATCH_INTERVAL" -eq 0 ]]; then
    break
  fi
  echo
  echo "Sleeping ${WATCH_INTERVAL}s; press Ctrl-C to stop."
  sleep "$WATCH_INTERVAL"
  echo
done
"""


def build_scripts() -> dict[str, str]:
    return {
        "gpu0_closure_contraction_pilot.sh": profile_script(
            gpu=0,
            title="closure/contraction pilot",
            name="pilot_closure_contraction_gpu0",
            datasets=["dyn_pubmed", "arxivmath"],
            batches=["999:50"],
            methods=["leidenalg"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            max_updates=8,
            timeout="6h",
        ),
        "gpu1_workload_memory_pilot.sh": profile_combo_script(),
        "gpu2_dsbm_update_size_pilot.sh": dsbm_script(
            gpu=2,
            title="DSBM update-size pilot",
            name="pilot_dsbm_update_size_gpu2",
            batch_suffix="5_batches",
            max_changes=[290, 1450],
            timeout="8h",
            limit=None,
        ),
        "gpu3_topology_variance_final.sh": launcher_script(
            3,
            "topology variance final",
            ["topology_variance_final.json"],
            timeout="24h",
        ),
        "gpu4_topology_long_horizon_final.sh": launcher_script(
            4,
            "topology long-horizon final",
            ["topology_long_horizon_final.json"],
            timeout="72h",
        ),
        "gpu5_s2cag_focused_final.sh": launcher_script(
            5,
            "S2CAG focused final",
            ["s2cag_naive_dataset_final.json", "s2cag_smart_random_final.json"],
            timeout="48h",
        ),
        "gpu6_closure_contraction_final.sh": profile_script(
            gpu=6,
            title="closure/contraction final",
            name="final_closure_contraction_gpu6",
            datasets=["dyn_pubmed", "arxivmath"],
            batches=["999:50"],
            methods=["leidenalg"],
            variants=["full", "no_closure", "no_contraction"],
            feature_modes=None,
            aggregation_mode="sum",
            baseline_iter=None,
            max_updates=50,
            timeout="24h",
        ),
        "gpu7_dsbm_update_size_final.sh": dsbm_script(
            gpu=7,
            title="DSBM update-size final",
            name="final_dsbm_update_size_gpu7",
            batch_suffix="10_batches",
            max_changes=[290, 1450, 2900],
            timeout="48h",
            limit=None,
        ),
        "monitor_cn69_jobs.sh": monitor_script(),
    }


README = """# cn69 Measurement Batch 2026-05-27

This batch restores the acceptance-critical direct closure/contraction
ablation and keeps placeholders only where a runnable measurement path exists.
It also matches the fixed experimental figure set in
`article/ICDM_REFACTOR_PLAN.md`.

Pilots:

- `gpu0_closure_contraction_pilot.sh`: direct `full/no_closure/no_contraction`
  Leiden pilot on `dyn_pubmed` and `arxivmath`; de-risks
  `tab:closure-ablation`.
- `gpu1_workload_memory_pilot.sh`: focused workload, contracted-size, timing,
  CPU RSS, and CUDA memory pilot for the 3 x 2 main grid; fills
  `tab:contracted-workload`, `tab:breakdown`, and `fig:workload-speedup`.
  It profiles `999:10` and a bounded prefix of `999:50` so the mechanism plot
  is not tied to a single short compatibility stream.
- `gpu2_dsbm_update_size_pilot.sh`: bounded synthetic update-size pilot across
  random, hub-centered, and community-internal regimes; de-risks
  `fig:update-size`.

Final/battle runs:

- `gpu3_topology_variance_final.sh`: fills topology focused-grid variance and
  contributes to `fig:quality-runtime-pareto`.
- `gpu4_topology_long_horizon_final.sh`: fills
  `fig:long-horizon-curves`.
- `gpu5_s2cag_focused_final.sh`: fills feature-aware focused rows and
  contributes to `fig:quality-runtime-pareto`.
- `gpu6_closure_contraction_final.sh`: final version of
  `tab:closure-ablation` after the gpu0 pilot is healthy.
- `gpu7_dsbm_update_size_final.sh`: fills `fig:update-size`.

The quality-runtime Pareto figure itself needs no separate GPU script: build it
from `results/registry/` after `gpu3` and `gpu5` are ingested.

Run pilots first if you want a conservative staged campaign. If the pilot logs
look healthy, run the corresponding final scripts. All shell logs go to
`output/`; JSON results go to `results/` or `results/paper_icdm/`.

Background container launch lines:

```bash
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu0_closure_contraction_pilot.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu1_workload_memory_pilot.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu2_dsbm_update_size_pilot.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu3_topology_variance_final.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu4_topology_long_horizon_final.sh'
docker exec -d dev_drobyshev bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu5_s2cag_focused_final.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu6_closure_contraction_final.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_pilot_20260527/gpu7_dsbm_update_size_final.sh'
```

After jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```

Monitor the eight background jobs from the host:

```bash
scripts/paper/cn69_pilot_20260527/monitor_cn69_jobs.sh
scripts/paper/cn69_pilot_20260527/monitor_cn69_jobs.sh --watch 60 --tail 3
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
