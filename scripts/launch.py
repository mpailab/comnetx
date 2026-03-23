import json
import sys
import os
from datetime import datetime
from itertools import product

if len(sys.argv[1:]) != 1:
  print("Usage: launch.py <config_file>")
  sys.exit(1)
conf_file = os.path.abspath(sys.argv[1])
if not os.path.exists(conf_file):
  print("Don't exists the file:", conf_file)
  sys.exit(1)
with open(conf_file) as _:
  conf = json.load(_)
conf_name = os.path.basename(conf_file).rsplit(".", maxsplit=1)[0]

# inner imports
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))
from launcher import dynamic_launch
from datasets import INFO, Dataset
from our_utils import print_zone

# input
MACHINE_DEFAULT = os.getenv('HOSTNAME')
MACHINE_PARENT = os.getenv('PARENT_HOSTNAME') # задать в bash : export PARENT_HOSTNAME=<parent_hostname>
MACHINE = MACHINE_PARENT if MACHINE_PARENT is not None else MACHINE_DEFAULT

# output
VERBOSE = conf.get("VERBOSE", 1) # 0, 1, 2, 3
CATCH_ERRORS = conf.get("CATCH_ERRORS", True)

#paths to datasets
paths_config_default = "datasets-info/paths/default.json"
paths_config_host = f"datasets-info/paths/{MACHINE}.json"
if os.path.exists(paths_config_host):
    paths_config = paths_config_host
else:
    print(f"Warning! {paths_config_host} does not exist.")
    print(f"         {paths_config_default} will be used instead.")
    paths_config = paths_config_default

# datasets
with open(os.path.join(INFO, "konect.json")) as _:
    info = json.load(_)
konect_datasets = list(filter(lambda dataset_name: info[dataset_name]["w"] in ["weighted", "unweighted"], list(info.keys())))
datasets_by_edges = sorted(konect_datasets, key = lambda x: info[x]["m"])
datasets_by_nodes = sorted(konect_datasets, key = lambda x: info[x]["n"])
undirected_datasets_by_nodes = list(filter(lambda dataset_name: info[dataset_name]["d"] == "undirected", datasets_by_nodes))
undirected_datasets_by_edges = list(filter(lambda dataset_name: info[dataset_name]["d"] == "undirected", datasets_by_edges))
datasets_dict = {
    "konect_by_edges" : datasets_by_edges,
    "konect_by_nodes" : datasets_by_nodes,
    "undirected_datasets_by_nodes" : undirected_datasets_by_nodes,
    "undirected_datasets_by_edges" : undirected_datasets_by_edges
}
if type(conf["DATASETS"]) == str:
  DATASETS = datasets_dict[conf["DATASETS"]]
elif type(conf["DATASETS"]) == list:
  DATASETS = conf["DATASETS"]
else:
  print('conf["DATASETS"] is not str or list:', conf["DATASETS"])
  sys.exit(1)
BATCHES = conf["BATCHES"] # [1, 10, 100, "real", "10:100"]

# algorithm
METHODS = conf["BASELINES"] # ["prgpt:locale", "prgpt:infomap", "leidenalg", "networkit", "magi", "dmon"]
MODES = conf["MODES"] # ["smart", "naive", "raw"]
SMART_VERSION = conf["SMART_VERSION"]
USE_GPU = conf.get("USE_GPU", True)

# baseline iterations
SUPPORTED_ITER_METHODS = {"magi", "dmon", "dese", "flmig", "s2cag"}
BASELINE_ITER_VALS = conf.get("BASELINE_ITERATIONS", [None])
if isinstance(BASELINE_ITER_VALS, (int, float)):
    BASELINE_ITER_VALS = [BASELINE_ITER_VALS]
elif not isinstance(BASELINE_ITER_VALS, list):
    BASELINE_ITER_VALS = [None]

SMART_PARAMS_GRID = conf.get("SMART_PARAMS_GRID", {})
# Пример SMART_PARAMS_GRID в конфиге:
# {
#    "smart_subcoms_depth": [3, 4, 5],
#    "smart_neighborhood_step": [1, 2, 3]
# }
SMART_PARAMS_LISTS = list(product(*SMART_PARAMS_GRID.values())) if SMART_PARAMS_GRID else [()]
ABBR = {"smart_subcoms_depth": "L", "smart_neighborhood_step": "r"}
REVERSE_ABBR = {v: k for k, v in ABBR.items()}
SMART_PAR_DEFAULT = {"smart_subcoms_depth": 5, "smart_neighborhood_step": 1}

def init(db, baseline, dataset_name):
    if baseline not in db:
        db[baseline] = {}
    if dataset_name not in db[baseline]:
        db[baseline][dataset_name] = {}
    if MACHINE not in db[baseline][dataset_name]:
        db[baseline][dataset_name][MACHINE] = {}
    return db

if conf.get("USE_TIMESTAMP_SUFFIX", True):
    DATE_SUFFIX = datetime.now().strftime('%Y%m%d_%H%M')
else:
    DATE_SUFFIX = "now"
def save(db, errors):
    os.makedirs(os.path.join(PROJECT_PATH, "results"), exist_ok = True)
    with open(os.path.join(PROJECT_PATH, "results", f"measurements_{conf_name}_{DATE_SUFFIX}.json"), 'w') as _:
        json.dump(db, _, indent=4)
    if errors:
        with open(os.path.join(PROJECT_PATH, "results", f"errors_{conf_name}_{DATE_SUFFIX}.json"), 'w') as _:
            json.dump(errors, _, indent=4)

def get_algname(method, mode, use_gpu, smart_params=None, baseline_iter=None):
    if method in SUPPORTED_ITER_METHODS and baseline_iter is not None:
        res = f"{method}-i:{baseline_iter}"
    else:
        res = method

    if mode == "smart" and smart_params:
        params_string = "-".join([f"{ABBR[k]}:{v}" for k, v in smart_params.items()])
        gpu_sfx = "gpu" if use_gpu else "cpu"
        res = f"{res}-{params_string}-{gpu_sfx}"
    else:
        res = f"{res}-{mode}"
        
    return res

def measure():
    db = {}
    errors = []

    for dataset_name in DATASETS:
        for batches_strategy in BATCHES:
            ds = Dataset(dataset_name, paths_config)
            ds.load(batches_strategy=batches_strategy)
            for method in METHODS:
                for mode in MODES:
                    if method in SUPPORTED_ITER_METHODS:
                        iter_vals = BASELINE_ITER_VALS
                    else:
                        iter_vals = [None]

                    for baseline_iter in iter_vals:
                        if mode == "smart" and SMART_PARAMS_GRID:
                            smart_params_list = SMART_PARAMS_LISTS
                        else:
                            smart_params_list = [()]

                        for smart_params_tuple in smart_params_list:
                            if smart_params_tuple:
                                keys = list(SMART_PARAMS_GRID.keys())
                                smart_params_dict = dict(zip(keys, smart_params_tuple))
                            else:
                                smart_params_dict = SMART_PAR_DEFAULT

                            algname = get_algname(method, mode, USE_GPU, smart_params_dict, baseline_iter)
                            db = init(db, algname, dataset_name)

                            try:
                                with print_zone(VERBOSE >= 1):
                                    print("-----------------------------------------------")
                                    print(f"Dataset: {dataset_name} ({batches_strategy} batches)")
                                    print(f"Baseline: {algname}")
                                results = dynamic_launch(
                                    ds,
                                    batches_strategy,
                                    method,
                                    baseline_iter=baseline_iter,
                                    mode=mode,
                                    smart_subcoms_depth=smart_params_dict["smart_subcoms_depth"],
                                    smart_neighborhood_step=smart_params_dict["smart_neighborhood_step"],
                                    verbose=VERBOSE,
                                    use_gpu=USE_GPU
                                )
                            except Exception as e:
                                if CATCH_ERRORS:
                                    err_tuple = (algname, dataset_name, batches_strategy, str(e))
                                    errors.append(err_tuple)
                                    print(f"Error {e} on:", dataset_name, batches_strategy, algname)
                                else:
                                    raise
                            else:
                                db[algname][dataset_name][MACHINE][str(batches_strategy)] = results
                                save(db, errors)
    return db, errors

if __name__ == "__main__":
    db, errors = measure()