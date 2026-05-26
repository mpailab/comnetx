import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from baselines.lago import lago_partition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--nb-iter", type=int, default=1)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    new_labels = lago_partition(
        adj,
        directed=args.directed,
        nb_iter=args.nb_iter,
    )
    torch.save(new_labels, args.out)
    print("LAGO finished successfully")


if __name__ == "__main__":
    main()
