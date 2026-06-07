#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

PIP=(python -m pip)
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    PIP=(sudo -H python -m pip)
fi

"${PIP[@]}" install --upgrade pip setuptools wheel
"${PIP[@]}" install -r .devcontainer/requirements-dev.txt

# torch-sparse wheels are published via PyG for x86_64, while aarch64
# falls back to a source build that needs access to the already-installed
# torch package during build-time.
PYG_TORCH_TAG="$(
python - <<'PY'
import torch

version = torch.__version__.split("+", 1)[0]
cuda = torch.version.cuda
backend = f"cu{cuda.replace('.', '')}" if cuda else "cpu"
print(f"torch-{version}+{backend}")
PY
)"

if ! python - <<'PY'
import torch_scatter
import torch_sparse
PY
then
    if [ "$(uname -m)" = "x86_64" ]; then
        "${PIP[@]}" install --force-reinstall --no-deps \
            torch-scatter -f "https://data.pyg.org/whl/${PYG_TORCH_TAG}.html"
        "${PIP[@]}" install --force-reinstall --no-deps \
            torch-sparse -f "https://data.pyg.org/whl/${PYG_TORCH_TAG}.html"
    else
        "${PIP[@]}" install --force-reinstall --no-cache-dir --no-build-isolation --no-deps torch-scatter
        "${PIP[@]}" install --force-reinstall --no-cache-dir --no-build-isolation --no-deps torch-sparse
    fi
fi

python - <<'PY'
import torch_scatter
import torch_sparse
PY

git config --global --add safe.directory /workspace

echo "[comnetx] Lightweight devcontainer dependencies installed."
echo "[comnetx] Heavy optional dependencies (graph-tool, local wheels, GPU stack, datasets) stay project-specific."
echo "[comnetx] Default Codex verification command: make verify"
