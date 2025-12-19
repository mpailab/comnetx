# install python3.12
# add-apt-repository ppa:deadsnakes/ppa
# apt-get update
# apt-get install python3.12
# apt install python3.12-full
# apt install python3-pip
# python3.12 -m ensurepip --upgrade
# python3.12 -m pip install --upgrade pip

# install python3-graph-tool
# apt install wget
# wget https://downloads.skewed.de/skewed-keyring/skewed-keyring_1.1_all_$(lsb_release -s -c).deb
# dpkg -i skewed-keyring_1.1_all_$(lsb_release -s -c).deb
# echo "deb [signed-by=/usr/share/keyrings/skewed-keyring.gpg] https://downloads.skewed.de/apt $(lsb_release -s -c) main" > /etc/apt/sources.list.d/skewed.list
# apt-get update
# apt-get install python3-graph-tool

set -e

echo "[INFO] Installing system dependencies..."

# --- Базовые утилиты ---
apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*


echo "[INFO] Installing Python dependencies..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REQUIREMENTS_PATH="$PROJECT_ROOT/baselines/PRGPT/requirements.txt"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd -P )"

git config --global --add safe.directory "$REPO_ROOT"

# install pybind for sdp-clustering
pip install --no-cache-dir pybind11

# install requirements for PRGPT
pip install --no-cache-dir -r "$REQUIREMENTS_PATH"

# install libraries for MAGI
pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    ogb matplotlib tensorflow==2.14.0 pytest debugpy scikit-learn-intelex 

pip install --no-cache-dir leidenalg networkit==2.8.7 gudhi seaborn POT eagerpy umap-learn

echo "[INFO] Done!"