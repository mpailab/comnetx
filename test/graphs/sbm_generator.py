import argparse

from sbm import generate_sbm_graph_universal, generate_temporal_sbm_graph_local

def main():
    parser = argparse.ArgumentParser(description="Generate SBM or Temporal SBM graphs.")

    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument("--k", type=int, required=True, help="Number of communities")
    parser.add_argument("--p_in", type=float, required=True, help="Probability of edge within community")
    parser.add_argument("--p_out", type=float, required=True, help="Probability of edge between communities")
    parser.add_argument("--directed", action='store_true', help="Generate directed graph")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--mode", type=str, choices=['static', 'batch', 'auto'], default='auto', help="Mode for generation")
    parser.add_argument("--max_degree", type=int, default=None, help="Hard upper bound on vertex degree")
    parser.add_argument("--auto_degree_factor", type=float, default=2.0, help="Multiplier for automatic max_degree")

    parser.add_argument("--temporal", action='store_true', help="Generate temporal SBM graph")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of time steps (for temporal SBM)")
    parser.add_argument("--drift_prob", type=float, default=0.01, help="Probability that a node changes community (for temporal SBM)")
    parser.add_argument("--change_frac", type=float, default=0.01, help="Fraction of E0 that we may change (adds + dels) each step (for temporal SBM)")
    parser.add_argument("--enable_add", action='store_true', help="Flag to allow edges' adds (for temporal SBM)")
    parser.add_argument("--enable_del", action='store_true', help="Flag to allow edges' dels (for temporal SBM)")
    args = parser.parse_args()

    if args.temporal:
        graph = generate_temporal_sbm_graph_local(
            n=args.n,
            k=args.k,
            p_in=args.p_in,
            p_out=args.p_out,
            n_steps=args.n_steps,
            drift_prob=args.drift_prob,
            directed=args.directed,
            seed=args.seed,
            graph_type='tsbm',
            change_frac=args.change_frac,
            enable_add=args.enable_add,
            enable_del=args.enable_del,
            max_degree=args.max_degree,
            auto_degree_factor=args.auto_degree_factor
        )
    else:
        graph = generate_sbm_graph_universal(
            n=args.n,
            k=args.k,
            p_in=args.p_in,
            p_out=args.p_out,
            directed=args.directed,
            seed=args.seed,
            mode=args.mode,
            graph_type='sbm',
            max_degree=args.max_degree,
            auto_degree_factor=args.auto_degree_factor

        )

    print("Graph generated successfully!")
    return graph

if __name__ == "__main__":
    main()
