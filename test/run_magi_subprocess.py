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
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    features = torch.load(args.features)

    new_labels = magi(adj, features, epochs=args.epochs, batchsize=args.batchsize)

    torch.save(new_labels, args.out)
    print("MAGI finished successfully")

if __name__ == "__main__":
    main()
