import argparse
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from baselines.rough_PRGPT import rough_prgpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--features", required=False)
    parser.add_argument("--refine", type=str, default="infomap")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    features = torch.load(args.features)

    new_labels = rough_prgpt(adj, refine=args.refine)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(new_labels, args.out)
    print("PRGPT finished successfully")


if __name__ == "__main__":
    main()
