import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import time

def generate_sbm_graph_universal(n, k, p_in, p_out, 
                       batch_size=None, directed=False, device='cpu', seed=None, 
                       ensure_connected=True, mode='auto', sparse_threshold=50000
):
    """
    Params:
        n : int nodes
        k : int communities
        p_in : float probability of edge within community
        p_out : float probability of edge between communities
        directed : bool
        device : str
        seed : int | None
        ensure_connected (bool): delete isolated nodes
        mode (str): 'static', 'batch' or 'auto' 
        sparse_threshold : int
            if n > sparse_threshold → create sparse matrix
    Returns:
        if mode='static' → (adj, labels)
        if mode='batch' → (adj_batches, label_batches)
    """
    rng = np.random.default_rng(seed)

    base_size = n // k
    block_sizes = [base_size] * k
    for i in range(n - base_size * k):
        block_sizes[i % k] += 1
    block_start = np.cumsum([0] + block_sizes[:-1])

    labels = np.concatenate([
        np.full(size, i, dtype=np.int64)
        for i, size in enumerate(block_sizes)
    ])
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    if mode == 'auto':
        if n > 100000 or (batch_size and batch_size < n):
            mode = 'batch'
        else:
            mode = 'static'

    use_sparse = n > sparse_threshold

    if mode == 'static':
        if use_sparse:
            rows, cols = [], []
            for i in range(k):
                for j in range(k):
                    p = p_in if i == j else p_out
                    u = np.arange(block_start[i], block_start[i] + block_sizes[i])
                    v = np.arange(block_start[j], block_start[j] + block_sizes[j])
                    mask = rng.random((len(u), len(v))) < p
                    ui, vj = np.nonzero(mask)
                    rows.extend(u[ui])
                    cols.extend(v[vj])
            if not directed:
                rows = np.concatenate([rows, cols])
                cols = np.concatenate([cols, rows[:len(cols)]])
            values = np.ones(len(rows), dtype=np.float32)
            adj_torch = torch.sparse_coo_tensor(
                torch.tensor([rows, cols]),
                torch.tensor(values),
                (n, n),
                device=device
            ).coalesce()
        else:
            A = np.zeros((n, n), dtype=np.float32)
            for i in range(k):
                for j in range(k):
                    p = p_in if i == j else p_out
                    ni = block_sizes[i]
                    nj = block_sizes[j]
                    u = np.arange(block_start[i], block_start[i] + ni)
                    v = np.arange(block_start[j], block_start[j] + nj)
                    sub = rng.random((ni, nj)) < p
                    A[np.ix_(u, v)] = sub.astype(np.float32)
            if not directed:
                A = np.triu(A, 1)
                A = A + A.T
            if ensure_connected:
                deg = A.sum(axis=1)
                isolated = np.where(deg == 0)[0]
                for v in isolated:
                    block_v = labels[v]
                    same_block = np.where(labels == block_v)[0]
                    same_block = same_block[same_block != v]
                    if len(same_block) > 0:
                        u = rng.choice(same_block)
                        A[v, u] = 1
                        if not directed:
                            A[u, v] = 1
            adj_torch = torch.tensor(A, dtype=torch.float32, device=device)

        return adj_torch, labels_t

    elif mode == 'batch':
        if batch_size is None:
            batch_size = 100

        def generate_batch(start, end):
            batch_n = end - start
            if use_sparse:
                rows, cols = [], []
                for i in range(k):
                    for j in range(k):
                        p = p_in if i == j else p_out
                        u_local = np.where(labels[start:end] == i)[0]
                        v_global = np.arange(block_start[j], block_start[j] + block_sizes[j])
                        if len(u_local) == 0 or len(v_global) == 0:
                            continue
                        mask = rng.random((len(u_local), len(v_global))) < p
                        ui, vj = np.nonzero(mask)
                        rows.extend(start + u_local[ui])
                        cols.extend(v_global[vj])
                if not directed:
                    rows = np.concatenate([rows, cols])
                    cols = np.concatenate([cols, rows[:len(cols)]])
                values = np.ones(len(rows), dtype=np.float32)
                adj = torch.sparse_coo_tensor(
                    torch.tensor([rows, cols]),
                    torch.tensor(values),
                    (n, n),
                    device=device
                ).coalesce()
            else:
                A = np.zeros((batch_n, n), dtype=np.float32)
                for i in range(k):
                    for j in range(k):
                        p = p_in if i == j else p_out
                        u_idx = np.where(labels[start:end] == i)[0]
                        v_idx = np.arange(block_start[j], block_start[j] + block_sizes[j])
                        if len(u_idx) == 0 or len(v_idx) == 0:
                            continue
                        sub = rng.random((len(u_idx), len(v_idx))) < p
                        A[np.ix_(u_idx, v_idx)] = sub.astype(np.float32)
                if not directed:
                    A[:, start:end] = np.triu(A[:, start:end], 1)
                    A[:, start:end] = A[:, start:end] + A[:, start:end].T
                adj = torch.tensor(A, dtype=torch.float32, device=device)
            return adj

        adj_batches, label_batches = [], []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            adj_batches.append(generate_batch(start, end))
            label_batches.append(labels_t[start:end])

        return adj_batches, label_batches

    else:
        raise ValueError(f"Error mode: {mode}")


def generate_temporal_sbm_graph(n, k, p_in, p_out, n_steps=10, drift_prob=0.01, 
                                edge_persistence=0.9, directed=False, device='cpu',
                                seed=None, ensure_connected=True
):
    """
    Params:
         n : int nodes
        k : int communities
        p_in : float probability of edge within community
        p_out : float probability of edge between communities
        n_steps (int): time steps
        drift_prob (float): probability that node change community
        edge_persistence (float): probability that existing edge will be preserved in the next step
        directed : bool
        device : str 'cpu' or 'cuda'
        seed : int | None
        ensure_connected (bool): delete isolated nodes
        mode (str): 'static', 'batch' or 'auto' 
        sparse_threshold : int
            if n > sparse_threshold → create sparse matrix
    Returns:
        temporal_adjs: list of tensors [T]
        temporal_labels: list of tensors [T]
    """
    rng = np.random.default_rng(seed)

    base_size = n // k
    block_sizes = [base_size] * k
    for i in range(n - base_size * k):
        block_sizes[i % k] += 1
    labels = np.concatenate([
        np.full(size, i, dtype=np.int64)
        for i, size in enumerate(block_sizes)
    ])

    def gen_adj(labels):
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(k):
            for j in range(k):
                p = p_in if i == j else p_out
                u = np.where(labels == i)[0]
                v = np.where(labels == j)[0]
                if len(u) == 0 or len(v) == 0:
                    continue
                sub = rng.random((len(u), len(v))) < p
                A[np.ix_(u, v)] = sub.astype(np.float32)
        if not directed:
            A = np.triu(A, 1)
            A = A + A.T
        if ensure_connected:
            deg = A.sum(axis=1)
            isolated = np.where(deg == 0)[0]
            for v in isolated:
                block_v = labels[v]
                same_block = np.where(labels == block_v)[0]
                same_block = same_block[same_block != v]
                if len(same_block) > 0:
                    u = rng.choice(same_block)
                    A[v, u] = 1
                    if not directed:
                        A[u, v] = 1
        return A

    A_prev = gen_adj(labels)
    temporal_adjs = [torch.tensor(A_prev, dtype=torch.float32, device=device)]
    temporal_labels = [torch.tensor(labels, dtype=torch.long, device=device)]

    for t in range(1, n_steps):
        drift_mask = rng.random(n) < drift_prob
        for i in np.where(drift_mask)[0]:
            old = labels[i]
            new = rng.choice([x for x in range(k) if x != old])
            labels[i] = new

        A_new = A_prev.copy()

        keep_mask = rng.random(A_prev.shape) < edge_persistence
        A_new = A_new * keep_mask

        for i in range(k):
            for j in range(k):
                p = p_in if i == j else p_out
                u = np.where(labels == i)[0]
                v = np.where(labels == j)[0]
                if len(u) == 0 or len(v) == 0:
                    continue
                sub_new = rng.random((len(u), len(v))) < p * (1 - edge_persistence)
                A_new[np.ix_(u, v)] = np.maximum(A_new[np.ix_(u, v)], sub_new.astype(np.float32))
        
        if not directed:
            A_new = np.triu(A_new, 1)
            A_new = A_new + A_new.T
        
        if ensure_connected:
            deg = A_new.sum(axis=1)
            isolated = np.where(deg == 0)[0]
            for v in isolated:
                block_v = labels[v]
                same_block = np.where(labels == block_v)[0]
                same_block = same_block[same_block != v]
                if len(same_block) > 0:
                    u = rng.choice(same_block)
                    A_new[v, u] = 1
                    if not directed:
                        A_new[u, v] = 1

        A_prev = A_new
        temporal_adjs.append(torch.tensor(A_new, dtype=torch.float32, device=device))
        temporal_labels.append(torch.tensor(labels.copy(), dtype=torch.long, device=device))

    return temporal_adjs, temporal_labels




start = time.time()
adj, labels = generate_sbm_graph_universal(
    n=200,
    k=4,
    p_in=0.08,
    p_out=0.005,
    directed=True,
    seed=42
)
print("Generation time:", round(time.time()-start, 3), "sec")
A_np = adj.cpu().numpy()
G = nx.from_numpy_array(A_np)
colors = [plt.cm.tab10(int(c)) for c in labels.cpu().numpy()]

plt.figure(figsize=(7, 7))
pos = nx.spring_layout(G, seed=42, k=0.15)
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=40, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
plt.title("Stochastic Block Model")
plt.axis('off')
plt.savefig("graph.png", dpi=150)

start = time.time()
temporal_adjs, temporal_labels = generate_temporal_sbm_graph(
    n=500,
    k=4,
    p_in=0.08,
    p_out=0.005,
    n_steps=10,
    drift_prob=0.02,         
    edge_persistence=0.9,    
    seed=42
)
print("Generation time for temporal:", round(time.time()-start, 3), "sec")
for t in [0, 3, 6, 9]:
    G = nx.from_numpy_array(temporal_adjs[t].cpu().numpy())
    labels_t = temporal_labels[t].cpu().numpy()
    colors = [plt.cm.tab10(l % 10 / 10) for l in labels_t]

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title(f"Step {t}")
    plt.axis("off")
    plt.savefig(f"graph_temp_{t}.png", dpi=150)
    plt.close()

