#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p logs

CVD="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a DEVICES <<< "$CVD"
if [[ "${#DEVICES[@]}" -lt 2 ]]; then
  echo "[ERROR] Need at least 2 devices in CUDA_VISIBLE_DEVICES, got: '${CVD}'" >&2
  exit 1
fi

GPU0="${DEVICES[0]}"
GPU1="${DEVICES[1]}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]:-}"; do kill -9 "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM

echo "[INFO] GPU=${GPU0} -> logs/ogbn-products.sage.gpu${GPU0}.log" >&2
CUDA_VISIBLE_DEVICES="${GPU0}" bash -lc "
  yes y | ${PYTHON_BIN} baselines/MAGI/train_sage.py \
    --runs 10 --dataset ogbn-products --batchsize 2048 --max_duration 60 \
    --kmeans_device cuda --kmeans_batch 300000 --hidden 1024,1024,256 --size 10,10,10 \
    --wt 20 --wl 4 --tau 0.9 --ns 0.1 --lr 0.01 --epochs 400 --wd 0 --dropout 0
" &> "logs/ogbn-products.sage.gpu${GPU0}.log" &
PIDS+=("$!")

echo "[INFO] GPU=${GPU1} -> logs/Computers.gcn.gpu${GPU1}.log" >&2
CUDA_VISIBLE_DEVICES="${GPU1}" bash -lc "
  ${PYTHON_BIN} baselines/MAGI/train_gcn.py \
    --runs 10 --dataset Computers --hidden 1024,512 \
    --wt 100 --wl 3 --tau 0.9 --ns 0.1 --lr 0.0005 --epochs 400 --wd 1e-3
" &> "logs/Computers.gcn.gpu${GPU1}.log" &
PIDS+=("$!")

wait "${PIDS[0]}"
wait "${PIDS[1]}"

trap - INT TERM
echo "[INFO] Done." >&2
