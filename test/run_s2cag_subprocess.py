import argparse
import torch
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from baselines.s2cag import s2cag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    
    parser.add_argument("--dataset", type=str, default="data")
    parser.add_argument("--T", type=int, default=15)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--fdim", type=int, default=0)
    parser.add_argument("--method", type=str, default="sub")
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--tau", type=int, default=50)   

    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    features = torch.load(args.features)
    labels = torch.load(args.labels)

    print("n_runs =", args.runs)

    new_labels = s2cag(adj, features, labels,
                       T=args.T, n_runs=args.runs, alpha=args.alpha, 
                       method=args.method, gamma=args.gamma, tau=args.tau)

    torch.save(new_labels, args.out)
    print("S2CAG finished successfully")

if __name__ == "__main__":
    main()

