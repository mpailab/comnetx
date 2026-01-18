import argparse
import torch
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from baselines.flmig import flmig_adopted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj = torch.load(args.adj)

    new_labels = flmig_adopted(adj=adj, Number_iter = 100,
                               Beta=0.5, max_rb=10, return_labels=True)

    torch.save(new_labels, args.out)
    print("FLMIG finished successfully")

if __name__ == "__main__":
    main()