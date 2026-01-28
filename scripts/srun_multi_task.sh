#!/bin/bash
set -euo pipefail

# Flag toggled when the user requests GPU allocation.
use_gpu=0
# Optional override for CPU cores per command.
custom_cpus_per_task=""
# Memory per task requested from Slurm in gigabytes (powers of two).
mem_per_task_gb=""
# Task name used for container paths.
task_name="dynamic-graphs"

# Short help shown when the user passes no arguments or an invalid count.
usage() {
  cat <<'USAGE'
Usage: srun_multi_task.sh [options] <cmd1> [cmd2 ... cmd8]

Options:
  --task NAME        task to benchmark (dynamic-graphs|comnetx, default: dynamic-graphs)
  --gpu              request one GPU per command (default: CPU only)
  --cpus-per-task N  override CPU cores per command (default: 128/num_cmds)
  --mem-per-task G   request 2^k GB per task (optional)
  -h, --help         show this message and exit

Run 1 to 8 commands in parallel using srun. Each argument is
executed with "bash -lc" so you can pass compound shell commands
as a single quoted string.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|--use-gpu)
      use_gpu=1
      shift
      ;;
    --no-gpu)
      use_gpu=0
      shift
      ;;
    --cpus-per-task)
      if [[ $# -lt 2 ]]; then
        echo "--cpus-per-task requires a value" >&2
        exit 1
      fi
      custom_cpus_per_task="$2"
      shift 2
      ;;
    --mem-per-task)
      if [[ $# -lt 2 ]]; then
        echo "--mem-per-task requires a value" >&2
        exit 1
      fi
      mem_per_task_gb="$2"
      shift 2
      ;;
    --task)
      if [[ $# -lt 2 ]]; then
        echo "--task requires a value" >&2
        exit 1
      fi
      task_name="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if (( $# == 0 )); then
  usage
  exit 1
fi

if (( $# > 8 )); then
  echo "Error: at most 8 commands are supported." >&2
  usage
  exit 1
fi

case "$task_name" in
  dynamic-graphs|comnetx)
    ;;
  *)
    echo "Error: --task must be one of: dynamic-graphs, comnetx" >&2
    exit 1
    ;;
esac

# List of commands we need to run; each index maps to SLURM_PROCID.
commands=("$@")
task_count=${#commands[@]}

# Each machine exposes 128 CPU cores; use that when CPU count is not overridden.
total_cores=128
if [[ -n "$custom_cpus_per_task" ]]; then
  if ! [[ "$custom_cpus_per_task" =~ ^[0-9]+$ ]]; then
    echo "Error: --cpus-per-task expects a positive integer" >&2
    exit 1
  fi
  if (( custom_cpus_per_task <= 0 )); then
    echo "Error: --cpus-per-task must be greater than zero" >&2
    exit 1
  fi
  cpus_per_task=$custom_cpus_per_task
else
  cpus_per_task=$(( total_cores / task_count ))
fi

mem_per_task_arg=""
mem_per_cpu_arg=""
if [[ -n "$mem_per_task_gb" ]]; then
  if ! [[ "$mem_per_task_gb" =~ ^[0-9]+$ ]]; then
    echo "Error: --mem-per-task expects an integer gigabyte value" >&2
    exit 1
  fi
  if (( mem_per_task_gb <= 0 )); then
    echo "Error: --mem-per-task must be greater than zero" >&2
    exit 1
  fi
  if (( (mem_per_task_gb & (mem_per_task_gb - 1)) != 0 )); then
    echo "Error: --mem-per-task must be a power of two (1,2,4,...)" >&2
    exit 1
  fi
  mem_per_task_arg="${mem_per_task_gb}G"
  mem_per_task_mb=$(( mem_per_task_gb * 1024 ))
  mem_per_cpu_mb=$(( (mem_per_task_mb + cpus_per_task - 1) / cpus_per_task ))
  if (( mem_per_cpu_mb % 1024 == 0 )); then
    mem_per_cpu_arg="$(( mem_per_cpu_mb / 1024 ))G"
  else
    mem_per_cpu_arg="${mem_per_cpu_mb}M"
  fi
fi

# Container settings reused in the srun command.
container_image="/scratch/bokovgv/${task_name}.sqsh"
container_workdir="/${task_name}"
container_mounts="/scratch/konovalovay/${task_name}:/${task_name},"\
"/scratch/bokovgv/datasets/graphs/konect:/${task_name}/datasets"

# Serialize the commands into bash array literal syntax so the inner
# bash process sees the same array with proper quoting.
commands_literal="($(printf ' %q' "${commands[@]}"))"

srun_args=(
  --nodes=1
  --ntasks="$task_count"
  --cpus-per-task="$cpus_per_task"
  --cpu-bind=cores
  --container-image="$container_image"
  --container-workdir="$container_workdir"
  --container-mounts="$container_mounts"
)
if [[ -n "$mem_per_task_arg" ]]; then
  srun_args+=(--mem-per-cpu="$mem_per_cpu_arg")
fi
if (( use_gpu )); then
  srun_args+=(--gpus="$task_count" --gpus-per-task=1)
fi

export SRUN_MULTI_USE_GPU="$use_gpu"

srun "${srun_args[@]}" \
     bash -lc '
       set -euo pipefail

       # Restore the serialized array with the original commands.
       declare -a CMDS='"$commands_literal"'
       proc_id=${SLURM_PROCID:-0}
       total=${#CMDS[@]}
       if (( proc_id >= total )); then
         echo "No command for SLURM_PROCID=$proc_id" >&2
         exit 1
       fi
       cmd="${CMDS[$proc_id]}"
       use_gpu="${SRUN_MULTI_USE_GPU:-0}"
       if (( use_gpu )); then
         # Keep this task on the GPU matching its Slurm proc id.
         export CUDA_VISIBLE_DEVICES="$proc_id"
       fi

       ts_fmt="+%Y-%m-%d %H:%M:%S"
       idx=$((proc_id + 1))
       log_line() {
         local text="$1"
         printf "[%s][task %d/%d] %s\n" "$(date "$ts_fmt")" \
                "$idx" "$total" "$text"
       }

       log_line "starting: $cmd"
       if bash -lc "$cmd"; then
         log_line "finished successfully"
       else
         rc=$?
         log_line "failed with exit code $rc" >&2
         exit "$rc"
       fi
     '
