import argparse
import torch
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from baselines.magi import magi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    features = torch.load(args.features)

    new_labels = adapted_dmon(adj, features, _n_epochs=args.epochs)
    print("MAGI finished successfully")

if __name__ == "__main__":
    main()
