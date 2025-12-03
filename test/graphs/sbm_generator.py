import argparse

from sbm import generate_sbm_graph_universal, generate_temporal_sbm_graph_local

def main():
    parser = argparse.ArgumentParser(description="Generate SBM or Temporal SBM graphs.")

    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument("--k", type=int, required=True, help="Number of communities")
    parser.add_argument("--p_in", type=float, required=True, help="Probability of edge within community")
    parser.add_argument("--p_out", type=float, required=True, help="Probability of edge between communities")
    parser.add_argument("--directed", action='store_true', help="Generate directed graph")
    parser.add_argument("--device", type=str, default='cpu', help="'cpu' or 'cuda'")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--ensure_connected", action='store_true', help="Ensure no isolated nodes")
    parser.add_argument("--mode", type=str, choices=['static', 'batch', 'auto'], default='auto', help="Mode for generation")

    parser.add_argument("--temporal", action='store_true', help="Generate temporal SBM graph")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of time steps (for temporal SBM)")
    parser.add_argument("--drift_prob", type=float, default=0.01, help="Probability that a node changes community (for temporal SBM)")
    parser.add_argument("--edge_persistence", type=float, default=0.9, help="Probability that an edge persists to next step (for temporal SBM)")
    parser.add_argument("--change_frac", type=float, default=0.01, help="Fraction of E0 that we may change (adds + dels) each step (for temporal SBM)")
    parser.add_argument("--enable_add", action='store_true', help="Flag to allow edges' adds (for temporal SBM)")
    parser.add_argument("--enable_del", action='store_true', help="Flag to allow edges' dels (for temporal SBM)")
    parser.add_argument("--use_ed_per", action='store_true', help="Each previous edge is kept with probability edge_persistence (for temporal SBM)")

    args = parser.parse_args()

    if args.temporal:
        graph = generate_temporal_sbm_graph_local(
            n=args.n,
            k=args.k,
            p_in=args.p_in,
            p_out=args.p_out,
            n_steps=args.n_steps,
            drift_prob=args.drift_prob,
            edge_persistence=args.edge_persistence,
            directed=args.directed,
            device=args.device,
            seed=args.seed,
            ensure_connected=args.ensure_connected,
            graph_type='tsbm',
            change_frac=args.change_frac,
            enable_add=args.enable_add,
            enable_del=args.enable_del, 
            use_edge_persistence=args.use_ed_per
        )
    else:
        graph = generate_sbm_graph_universal(
            n=args.n,
            k=args.k,
            p_in=args.p_in,
            p_out=args.p_out,
            directed=args.directed,
            device=args.device,
            seed=args.seed,
            ensure_connected=args.ensure_connected,
            mode=args.mode,
            graph_type='sbm'
        )

    print("Graph generated successfully!")
    return graph

if __name__ == "__main__":
    main()
