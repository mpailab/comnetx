import json
import subprocess
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
from datasets import INFO, KONECT_PATH

# input
MACHINE_DEFAULT = subprocess.check_output("hostname", shell=True, text=True)[:-1]
MACHINE = conf.get("MACHINE", MACHINE_DEFAULT)

# output
VERBOSE = conf.get("VERBOSE", 1) # 0, 1, 2, 3

# datasets
with open(os.path.join(INFO, "konect.json")) as _:
    info = json.load(_)
konect_datasets = list(filter(lambda dataset: info[dataset]["w"] in ["weighted", "unweighted"], list(info.keys())))
datasets_by_edges = sorted(konect_datasets, key = lambda x: info[x]["m"])
datasets_by_nodes = sorted(konect_datasets, key = lambda x: info[x]["n"])
datasets_dict = {
    "konect_by_edges" : datasets_by_edges,
    "konect_by_nodes" : datasets_by_nodes
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

def init(db, baseline, dataset):
    if baseline not in db:
        db[baseline] = {}
    if dataset not in db[baseline]:
        db[baseline][dataset] = {}
    if MACHINE not in db[baseline][dataset]:
        db[baseline][dataset][MACHINE] = {}
    return db

#DATE_SUFFIX = datetime.now().strftime('%Y%m%d_%H%M')
DATE_SUFFIX = "now"
def save(db, errors):
    os.makedirs(os.path.join(PROJECT_PATH, "results"), exist_ok = True)
    with open(os.path.join(PROJECT_PATH, "results", f"measurements_{DATE_SUFFIX}.json"), 'w') as _:
        json.dump(db, _, indent=4)
    with open(os.path.join(PROJECT_PATH, "results", f"errors_{DATE_SUFFIX}.json"), 'w') as _:
        #print(errors)
        json.dump(errors, _, indent=4)

def get_algname(method, mode, smart_params=None):
    if mode == "smart":
        #algname = f"{method}-{SMART_VERSION}"
        algname = f"{method}"
        if smart_params:
            params_string = "-".join([f"{ABBR[k]}:{v}" for k, v in smart_params.items()])
            algname = f"{algname}-{params_string}"
        return algname
    else:
        return f"{method}-{mode}"

def measure():
    db = {}
    errors = []
    for dataset in DATASETS:
        for batches_num in BATCHES:
            for method in METHODS:
                for mode in MODES:
                    if mode == "smart" and SMART_PARAMS_GRID:
                        smart_params_list = SMART_PARAMS_LISTS
                    else:
                        smart_params_list = [()]  # один пустой набор для не-smart режимов
                    
                    for smart_params_tuple in smart_params_list:
                        if smart_params_tuple:
                            keys = list(SMART_PARAMS_GRID.keys())
                            smart_params_dict = dict(zip(keys, smart_params_tuple))
                        else:
                            smart_params_dict = SMART_PAR_DEFAULT
                        
                        algname = get_algname(method, mode, smart_params_dict)
                        db = init(db, algname, dataset)
                        
                        try:
                            results = dynamic_launch(
                                dataset, 
                                batches_num,
                                method, 
                                mode=mode,
                                smart_subcoms_depth=smart_params_dict["smart_subcoms_depth"],
                                smart_neighborhood_step=smart_params_dict["smart_neighborhood_step"],
                                verbose=VERBOSE
                            )
                        except Exception as e:
                            err_tuple = (algname, dataset, batches_num, str(e))
                            errors.append(err_tuple)
                            print(f"Error {e} on:", dataset, batches_num, algname)
                        else:
                            db[algname][dataset][MACHINE][str(batches_num)] = results
                            save(db, errors)
    return db, errors

if __name__ == "__main__":
    db, errors = measure()