import argparse
import os
import sys
import time

import torch

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAGO_ROOT = os.path.join(PROJECT_PATH, "baselines", "LAGO")

if LAGO_ROOT not in sys.path:
    sys.path.insert(0, LAGO_ROOT)

from lago import LinkStream, lago_modules


def _normalize_refinement(refinement):
    if refinement is None:
        return None
    value = str(refinement).strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value.upper()


def _collect_edges_from_sparse(adj: torch.Tensor, directed: bool):
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    edge_weights = {}

    if adj.ndim == 2:
        iterator = zip(
            torch.zeros(indices.size(1), dtype=torch.long),
            indices[0],
            indices[1],
            values,
        )
    elif adj.ndim == 3:
        iterator = zip(indices[0], indices[1], indices[2], values)
    else:
        raise ValueError(f"Unsupported adjacency ndim for LAGO: {adj.ndim}")

    for time_idx, source, target, weight in iterator:
        weight_value = float(weight.item())
        if weight_value <= 0:
            continue

        t = int(time_idx.item())
        u = int(source.item())
        v = int(target.item())
        if not directed and u > v:
            u, v = v, u

        key = (t, u, v)
        if directed:
            edge_weights[key] = edge_weights.get(key, 0.0) + weight_value
        else:
            edge_weights[key] = max(edge_weights.get(key, 0.0), weight_value)

    return edge_weights


def _collect_edges_from_dense(adj: torch.Tensor, directed: bool):
    edge_weights = {}

    if adj.ndim == 2:
        nonzero = torch.nonzero(adj, as_tuple=False)
        for source, target in nonzero:
            weight_value = float(adj[source, target].item())
            if weight_value <= 0:
                continue
            u = int(source.item())
            v = int(target.item())
            if not directed and u > v:
                u, v = v, u
            key = (0, u, v)
            edge_weights[key] = (
                edge_weights.get(key, 0.0) + weight_value
                if directed
                else max(edge_weights.get(key, 0.0), weight_value)
            )
    elif adj.ndim == 3:
        nonzero = torch.nonzero(adj, as_tuple=False)
        for time_idx, source, target in nonzero:
            weight_value = float(adj[time_idx, source, target].item())
            if weight_value <= 0:
                continue
            t = int(time_idx.item())
            u = int(source.item())
            v = int(target.item())
            if not directed and u > v:
                u, v = v, u
            key = (t, u, v)
            edge_weights[key] = (
                edge_weights.get(key, 0.0) + weight_value
                if directed
                else max(edge_weights.get(key, 0.0), weight_value)
            )
    else:
        raise ValueError(f"Unsupported adjacency ndim for LAGO: {adj.ndim}")

    return edge_weights


def _adjacency_to_linkstream(adj: torch.Tensor, directed: bool):
    if adj.device.type == "cuda":
        adj = adj.cpu()
    if adj.layout != torch.strided and not adj.is_sparse:
        adj = adj.to_sparse_coo()

    edge_weights = (
        _collect_edges_from_sparse(adj, directed)
        if adj.is_sparse
        else _collect_edges_from_dense(adj, directed)
    )

    links = [
        (source, target, time_idx, weight)
        for (time_idx, source, target), weight in sorted(edge_weights.items())
    ]

    linkstream = LinkStream(directed=directed)
    if links:
        linkstream.add_links(links)
    return linkstream


def _modules_to_partition(modules, num_nodes: int, target_time: int):
    labels = torch.full((num_nodes,), -1, dtype=torch.long)

    membership = modules.get_nodes_modules_membership_at_time(target_time)
    for node, module in membership.items():
        if 0 <= int(node) < num_nodes:
            labels[int(node)] = int(module)

    next_label = (max(modules.modules) + 1) if modules.modules else 0
    for node in range(num_nodes):
        if labels[node] >= 0:
            continue

        trajectory = modules.get_node_trajectory(node)
        if trajectory:
            previous_times = [t for t in trajectory if t <= target_time]
            chosen_time = max(previous_times) if previous_times else max(trajectory)
            labels[node] = int(trajectory[chosen_time])
        else:
            labels[node] = next_label
            next_label += 1

    _, compact_labels = torch.unique(labels, sorted=True, return_inverse=True)
    return compact_labels.to(torch.long)


def lago_partition(
    adj: torch.Tensor,
    init_partition=None,
    timing_info=None,
    directed: bool = False,
    lex: str = "MM",
    nb_iter: int | None = None,
    gamma: float = 1.0,
    omega: float = 2.0,
    refinement=None,
    fast_exploration: bool = True,
    refinement_in: bool = True,
    seed: int | None = 42,
):
    """
    Run LAGO on a static adjacency matrix or a temporal adjacency tensor.

    ``init_partition`` is accepted for Optimizer compatibility. The upstream
    LAGO implementation does not expose a warm-start API, so it is intentionally
    ignored.
    """
    del init_partition

    if nb_iter is None:
        nb_iter = 1

    num_nodes = int(adj.size(-1))
    target_time = int(adj.size(0) - 1) if adj.ndim == 3 else 0

    conversion_start = time.perf_counter()
    if adj.device.type == "cuda":
        adj = adj.cpu()
    linkstream = _adjacency_to_linkstream(adj, directed=directed)
    conversion_time = time.perf_counter() - conversion_start
    if timing_info is not None:
        timing_info["conversion_time"] = (
            timing_info.get("conversion_time", 0.0) + conversion_time
        )

    if linkstream.nb_edges == 0:
        return torch.arange(num_nodes, dtype=torch.long)

    algorithm_start = time.perf_counter()
    modules = lago_modules(
        linkstream,
        lex=lex,
        nb_iter=int(nb_iter),
        gamma=float(gamma),
        omega=float(omega),
        refinement=_normalize_refinement(refinement),
        fast_exploration=fast_exploration,
        refinement_in=refinement_in,
        verbose=0,
        seed=seed,
    )
    algorithm_time = time.perf_counter() - algorithm_start
    if timing_info is not None:
        timing_info["algorithm_time"] = (
            timing_info.get("algorithm_time", 0.0) + algorithm_time
        )

    return _modules_to_partition(modules, num_nodes=num_nodes, target_time=target_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--lex", default="MM")
    parser.add_argument("--nb-iter", "--iter", dest="nb_iter", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=2.0)
    parser.add_argument("--refinement", default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    labels = lago_partition(
        adj,
        directed=args.directed,
        lex=args.lex,
        nb_iter=args.nb_iter,
        gamma=args.gamma,
        omega=args.omega,
        refinement=args.refinement,
        seed=args.seed,
    )
    torch.save(labels, args.out)
    print("LAGO finished successfully")


if __name__ == "__main__":
    main()
