import json
import subprocess
import sys
import os
from datetime import datetime

SMART_VERSION = "0.1"

MACHINE = subprocess.check_output("hostname", shell=True, text=True)[:-1]
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from launcher import dynamic_launch
from datasets import INFO

with open(os.path.join(INFO, "all.json")) as _:
    info = json.load(_)
DATASETS = list(filter(lambda dataset: info[dataset]["w"] in ["weighted", "unweighted"], list(info.keys())))

EDGE_RESTRICTION = 100000
#NODE_RESTRICTION = None
if EDGE_RESTRICTION:
    DATASETS = list(filter(lambda dataset: int(info[dataset]["m"]) < EDGE_RESTRICTION, DATASETS))
DATASETS.sort(key = lambda x: info[x]["n"])
BATCHES = [1, 10, 100]
#BATCHES = [1, 10, 100, 1000]
print("Number of datasets: ", len(DATASETS))

METHODS = ["prgpt:locale", "prgpt:infomap", "leidenalg"]
#METHODS = ["prgpt:locale", "prgpt:infomap", "leidenalg", "networkit", "magi", "dmon"]

# def get_true_modularities():
#     with open(os.path.join(INFO, "modularity.json")) as _:
#         true_mod = json.load(_)
#     return true_mod

def init(db, baseline, dataset):
    if baseline not in db:
        db[baseline] = {}
    if dataset not in db[baseline]:
        db[baseline][dataset] = {}
    if MACHINE not in db[baseline][dataset]:
        db[baseline][dataset][MACHINE] = {}
    return db

DATE_SUFFIX = datetime.now().strftime('%Y%m%d_%H%M')
def save(db, errors):
    os.makedirs(os.path.join(PROJECT_PATH, "results"), exist_ok = True)
    with open(os.path.join(PROJECT_PATH, "results", f"measurements_{DATE_SUFFIX}.json"), 'w') as _:
        json.dump(db, _, indent=4)
    with open(os.path.join(PROJECT_PATH, "results", f"errors_{DATE_SUFFIX}.json"), 'w') as _:
        json.dump(errors, _, indent=4)

def measure():
    db = {}
    errors = []
    for dataset in DATASETS:
        for batches_num in BATCHES:
            for method in METHODS:
                for mode in ["smart", "naive", "raw"]:
                    baseline = f"{method}-{mode}"
                    baseline = f"{baseline}-{SMART_VERSION}" if mode == "smart" else baseline
                    db = init(db, baseline, dataset)
                    try:
                        results = dynamic_launch(dataset, batches_num, method, mode = mode)
                    except Exception as e:
                        err_tuple = (baseline, dataset, batches_num, e)
                        errors.append(err_tuple)
                        print(f"Error {e} on:", dataset, batches_num, baseline)
                    else:
                        db[baseline][dataset][MACHINE][str(batches_num)] = results
                        save(db, errors)
    return db, errors

if __name__ == "__main__":
    db, errors = measure()
    
