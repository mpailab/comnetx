import json

PATH_OUR = "our_results.json"
PATH_NAIVE = "naive_results.json"
PATH_GPU = "gpu_results.json"

DATASETS = []
BATCHES = [1, 10, 100]
STATIC_BASELINES = []
GPU_DYNAMIC_BASELINES = []

def init_database(db, baseline, dataset):
    if baseline not in db:
        db[baseline] = {}
    if dataset not in db[baseline]:
        db[baseline][dataset] = {}
    return db

def dynamic_our_vs_naive():
    our_db = {}
    naive_db = {}
    gpu_db = {}

    for dataset in DATASETS:
        for batches_num in BATCHES:
            for baseline in STATIC_BASELINES:
                try:
                    local_our_db = dynamic_launch(dataset, batches_num, baseline, mode = "smart")
                    local_naive_db = dynamic_launch(dataset, batches_num, baseline, mode = "naive")
                except Exception as e:
                    print(f"Error {e} on:", dataset, batches_num, baseline)
                    continue

                our_db = init_database(our_db, baseline, dataset)
                naive_db = init_database(naive_db, baseline, dataset)

                our_db[baseline][dataset][batches_num] = local_our_db
                naive_db[baseline][dataset][batches_num] = local_naive_db

            for baseline in GPU_DYNAMIC_BASELINES:
                try:
                    local_gpu_db = dynamic_launch(dataset, batches_num, baseline, mode = "original")
                except Exception as e:
                    print(f"Error {e} on:", dataset, batches_num, baseline)
                    continue

                gpu_db = init_database(gpu_db, baseline, dataset)
                gpu_db[baseline][dataset][batches_num] = local_gpu_db
    
    with open(PATH_OUR, 'w') as f:
        json.dump(our_db, f, indent=4)
    with open(PATH_NAIVE, 'w') as f:
        json.dump(naive_db, f, indent=4)
    with open(PATH_GPU, 'w') as f:
        json.dump(gpu_db, f, indent=4)

# Note
# static_scenario(dataset, baseline) is equal to naive_dynamic_launch(dataset, 1, baseline)

def graphics_and_tables():
    our_db = json.load(PATH_OUR)
    naive_db = json.load(PATH_NAIVE)
    gpu_db = json.load(PATH_GPU)
    pass # Use scripts from the last project

if __name__ == "__main__":
    dynamic_our_vs_naive()
