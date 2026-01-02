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

# SBM generation params


To generate sbm datasets run `sbm_generator` with args:

## General settings
- `--n`, `type=int` — Number of nodes
- `--k`, `type=int` — Number of communities
- `--p_in`, `type=float` — Probability of edge within community
- `--p_out`, `type=float` —  Probability of edge between communities
- `--directed` — Generate directed graph
- `--device`, `type=str` — 'cpu' or 'cuda'
- `--seed`, `type=int` — Random seed
- `--ensure_connected` — Ensure no isolated nodes
- `--mode`, `type=str`, `choices=['static', 'batch', 'auto']` — Mode for generation (for static SBM graph)

## temporal SBM
- `--temporal` — Generate temporal SBM graph
- `--n_steps`, `type=int` —  Number of time steps
- `--drift_prob`, `type=float` — Probability that a node changes community
- `--edge_persistence`, `type=float` — Probability that an edge persists to next step
- `--change_frac`, `type=float` — Fraction of E0 that we may change (adds + dels) each step
- `--enable_add` — Flag to allow edges' adds
- `--enable_del` — Flag to allow edges' dels
- `--use_ed_per` — Flag for update setting:
  - `True` — each previous edge is kept with probability `edge_persistence`;  additions then fill up to `change_frac` budget
  - `False` — remove exactly `num_del_target` (if `enable_del`) and add `num_add_target` (if `enable_add`)
