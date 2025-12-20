import sys
import os
import pytest
import json

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from launcher import dynamic_launch
from datasets import INFO

datasets_1 = ['wiki_talk_ht', 'wiki_talk_cy', 'wiki_talk_oc', 'wiki_talk_br', 'sociopatterns-infectious', 'sociopatterns-hypertext']
datasets_2 = ['wiki_talk_nds', 'contact', 'dnc-temporalGraph', 'wiki_talk_eo', 'dblp-cite', 'opsahl-ucsocial', 'wiki_talk_gl']
datasets_3 = ['radoslaw_email', 'munmun_digg_reply', 'topology']
bad_datasets = ['wiki_talk_el', 'slashdot-threads', 'wiki_talk_sk', 'wiki_talk_bn', 'wiki_talk_lv', 'wiki_talk_eu', 'facebook-wosn-links', 'wiki_talk_sr']
datasets = datasets_1 + datasets_2 + datasets_3

@pytest.mark.short
def test_modularity():
    with open(os.path.join(INFO, "modularity.json")) as _:
         true_mod_db = json.load(_)
    with open(os.path.join(INFO, "konect.json")) as _:
        info = json.load(_)
    # datasets = list(filter(lambda dataset: info[dataset]["w"] in ["weighted", "unweighted"], list(info.keys())))
    # restriction = 100000
    # datasets = list(filter(lambda dataset: int(info[dataset]["m"]) < restriction, datasets))
    # datasets.sort(key = lambda x: info[x]["m"])
    datasets = datasets_1
    
    print("Dataset (is_directed) : Modularity (True modularity)")
    for dataset in datasets:
        results = dynamic_launch(dataset, 1, "leidenalg", mode = "naive", verbose = 0)
        mod = results[-1]['modularity']
        true_mod = true_mod_db[dataset]
        is_directed = info[dataset]["d"]
        print(f"{dataset} ({is_directed}): {mod:.4} (true - {true_mod:.4})")
        assert mod < true_mod * 1.1
        assert mod > true_mod * 0.9

@pytest.mark.short
def test_accuracy():
    with open(os.path.join(INFO, "konect.json")) as _:
        info = json.load(_)

    datasets = datasets_1
    
    for dataset in datasets:
        results = dynamic_launch(dataset, 1, "s2cag", mode = "naive", verbose = 0)
        acc = results[-1]['accuracy']
        print(f"{dataset} : {acc:.4}")
        assert acc < 0.0
        assert acc > 1.0

@pytest.mark.long    
def test_modularity_memory():
    datasets = ['wiki_talk_sr', 'sx-mathoverflow', 'wiki_talk_sv', 'wiki_talk_vi', 'facebook-wosn-links', 'facebook-wosn-wall', 'sx-askubuntu', 'wiki_talk_ja', 'mit', 'lkml-reply']
    for dataset in datasets:
        print(dataset)
        results = dynamic_launch(dataset, 1, "leidenalg", mode = "raw", verbose = 0)   
