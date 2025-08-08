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

echo "[INFO] Installing dependencies..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

REQUIREMENTS_PATH="$PROJECT_ROOT/baselines/PRGPT/requirements.txt"

# new
pip install pybind11 # for sdp-clustering !!!
pip install sdp-clustering
pip install -r "$REQUIREMENTS_PATH"
# torch.__version__  вместо 2.1.0+cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install ogb matplotlib pytest tensorflow debugpy 

echo "[INFO] Done!"