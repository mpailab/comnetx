import sys, json
from pathlib import Path
import faulthandler, signal

faulthandler.register(signal.SIGUSR1, all_threads=True)
TEST_PATH = Path(__file__).resolve().parent
PROJECT_PATH = TEST_PATH.parent

for p in (PROJECT_PATH, PROJECT_PATH / "src", TEST_PATH):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from launcher import dynamic_launch
 
if __name__ == "__main__":
    dataset, batches, method, mode, verbose = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
    results = dynamic_launch(dataset, batches, method, mode=mode, verbose=verbose)
    print(json.dumps({"results": results}))