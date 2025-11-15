# Test Graphs Information

# Small graphs:
| Dataset   | Samples | Dimension | Edges | Classes |
|-----------|--------:|----------:|------:|--------:|
| CORA      |    2708 |      1433 |  5278 |       7 |
| CITESEER  |    3327 |      3703 |  4552 |       6 |
| DBLP      |    4057 |       334 |  3528 |       4 |
| ACM       |    3025 |      1870 | 13128 |       3 |
| BAT       |     131 |        81 |  1038 |       4 |
| EAT       |     399 |       203 |  5994 |       4 |
| UAT       |    1190 |       239 | 13599 |       4 |

# SBM generation
To generate sbm datasets run sbm_generator with args:
--n, type=int, Number of nodes
--k, type=int, Number of communities
--p_in, type=float, Probability of edge within community
--p_out, type=float, Probability of edge between communities
--directed, Generate directed graph
--device", type=str, 'cpu' or 'cuda'
--seed, type=int, Random seed
--ensure_connected, Ensure no isolated nodes
--mode, type=str, choices=['static', 'batch', 'auto'], Mode for generation static SBM graph

--temporal, Generate temporal SBM graph
--n_steps, type=int, Number of time steps (for temporal SBM)
--drift_prob, type=float, Probability that a node changes community (for temporal SBM)
--edge_persistence, type=float, Probability that an edge persists to next step (for temporal SBM)

