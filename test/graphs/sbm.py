import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import math
import time
import os

def check_snapshot_symmetric(adj_3d, t):
    """
    adj_3d: sparse COO tensor of shape (T, n, n)
    t: snapshot index

    Returns True if symmetric, False otherwise.
    """

    # all indices of 3D tensor
    idx = adj_3d.coalesce().indices()      # shape (3, E)
    val = adj_3d.coalesce().values()       # shape (E,)

    # select entries where snapshot index == t
    mask = (idx[0] == t)
    idx2 = idx[:, mask]         # (3, E_t)
    val2 = val[mask]

    # extract u, v
    u = idx2[1].cpu().numpy()
    v = idx2[2].cpu().numpy()
    # keep only edges where value is positive (should be 1)
    good = val2.cpu().numpy() > 0
    u = u[good]; v = v[good]

    # build set of edges
    edges = set(zip(u, v))
    mirror = set(zip(v, u))

    return edges == mirror

def save_sbm_graph(rows, cols, n, labels, fname, meta=None, temporal=False, directed=False, connected=True):
    """
    Save graph (edge list) and labels in .pt file
    - rows, cols: int (index of nodes in edges)
    - labels: numpy array or torch tensor 
    - meta: (block_sizes, k, p_in/p_out)
    """
    os.makedirs("sbm", exist_ok=True)
    path = os.path.join("sbm", fname)

    meta_full = meta.copy() if meta else {}
    meta_full.update({
        "directed": directed,
        "connected": connected,
        "temporal": temporal,
        "n": n,
    })

    data = {"meta": meta_full}

    if temporal:
        assert isinstance(rows, list) and isinstance(cols, list), \
            "rows/cols must be list of lists in temporal-graph"
        data['rows'] = [np.asarray(r, dtype=np.int64) for r in rows]
        data['cols'] = [np.asarray(c, dtype=np.int64) for c in cols]
        data['labels'] = [torch.tensor(l, dtype=torch.long) for l in labels]
    else:
        data['rows'] = np.asarray(rows, dtype=np.int64)
        data['cols'] = np.asarray(cols, dtype=np.int64)
        data['labels'] = torch.tensor(labels, dtype=torch.long)

    torch.save(data, path)
    return path

def load_sbm_graph(fname, device='cpu'):
    """
    Load .pt file, returns:
      - adj_sparse: torch.sparse_coo_tensor (shape n x n), coalesced, dtype=torch.float32
      - labels_t: torch.tensor(labels, dtype=torch.long)
    """
    if not os.path.isabs(fname) and not os.path.exists(fname):
        candidate = os.path.join(os.getcwd(), "sbm", fname)
        if os.path.exists(candidate):
            fname = candidate

    data = torch.load(fname, map_location='cpu')  # load on cpu
    rows = np.asarray(data["rows"], dtype=np.int64)
    cols = np.asarray(data["cols"], dtype=np.int64)
    n = int(data["meta"]["n"])
    if rows.size == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
    else:
        indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
        values = torch.ones(indices.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce().to(device)
    labels_t = torch.tensor(data["labels"], dtype=torch.long, device=device)
    return adj, labels_t

def load_temporal_sbm_graph(path, device='cpu'):
    """
    Load temporal SBM graph as a 3D sparse tensor (T, n, n).
    Labels tensor - 2D (T, n).
    For undirected graphs we mirror per-snapshot by summing with the transpose
    """
    data = torch.load(path, map_location='cpu')
    meta = data.get('meta', {})
    if not meta.get('temporal', False):
        raise ValueError("Not a temporal SBM file")

    rows_list = data['rows']
    cols_list = data['cols']
    labels_list = data['labels']

    T = len(rows_list)
    n = int(meta['n'])
    directed = meta.get('directed', False)

    adjs = []
    for t, (r_arr, c_arr) in enumerate(zip(rows_list, cols_list)):
        r = np.asarray(r_arr, dtype=np.int64)
        c = np.asarray(c_arr, dtype=np.int64)
        e = len(r)
        if e == 0:
            adj_t = torch.sparse_coo_tensor(
                torch.empty((2,0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
                size=(n, n), dtype=torch.float32, device=device
            ).coalesce()
            adjs.append(adj_t)
            continue

        # remove duplicates in this snapshot (triples are (u,v))
        triples = np.vstack([r, c]).T
        triples = np.unique(triples, axis=0)

        # build indices for 2D snapshot
        indices2 = torch.from_numpy(triples.T).long().to(device)
        values2 = torch.ones(indices2.shape[1], dtype=torch.float32, device=device)

        adj_sparse = torch.sparse_coo_tensor(indices2, values2, size=(n, n), dtype=torch.float32, device=device).coalesce()

        if not directed:
            # mirror using transpose (create symmetric adjacency)
            # adj_t: same indices but flipped rows<->cols
            adj_t = torch.sparse_coo_tensor(adj_sparse.indices().flip(0), adj_sparse.values(), size=adj_sparse.shape, dtype=adj_sparse.dtype, device=device)
            adj_sparse = (adj_sparse + adj_t).coalesce()

        # ensure coalesced
        adj_sparse = adj_sparse.coalesce()
        adjs.append(adj_sparse)

    # stack into (T, n, n)
    adj_3d = torch.stack(adjs, dim=0)

    # labels -> tensor (T, n)
    labels = torch.stack([
        torch.from_numpy(l).long().to(device) if isinstance(l, np.ndarray) else l.clone().detach().long().to(device)
        for l in labels_list
    ], dim=0)

    return adj_3d, labels

def generate_sbm_graph_universal(n, k, p_in, p_out, batch_size=None, directed=False, seed=None,
                                 mode='auto', graph_type='sbm',
                                 max_degree=None, auto_degree_factor=2.0):
    """ 
    Params: 
    n : int nodes 
    k : int communities 
    p_in : float probability of edge within community 
    p_out : float probability of edge between communities 
    directed : bool 
    seed : int | None 
    mode (str): 'static', 'batch' or 'auto' 
    max_degree : int | None hard upper bound on vertex degree 
    auto_degree_factor : float multiplier for automatic max_degree 
    Returns: if mode='static' (adj, labels) 
            if mode='batch' (adj_batches, label_batches) 
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Blocks
    # --------------------------------------------------
    base = n // k
    block_sizes = [base] * k
    for i in range(n - base * k):
        block_sizes[i % k] += 1

    block_start = np.cumsum([0] + block_sizes[:-1])

    labels = np.concatenate([np.full(sz, i, dtype=np.int32)
                             for i, sz in enumerate(block_sizes)])
    # --------------------------------------------------
    # Automatic max_degree
    # --------------------------------------------------
    if max_degree is None:
        exp_deg = []
        for i in range(k):
            ni = block_sizes[i]
            exp_deg.append((ni - 1) * p_in + (n - ni) * p_out)
        max_degree = int(auto_degree_factor * max(exp_deg))
        if max_degree <= 0:
            max_degree = None

    # --------------------------------------------------
    # Mode
    # --------------------------------------------------
    if mode == 'auto':
        mode = 'batch' if (n >= 10000 or (batch_size and batch_size < n)) else 'static'

    batches_field = batch_size if (mode == 'batch' and batch_size is not None) else 0
    fname = (
        f"{graph_type}_{batches_field}b_{n}v_{k}c_"
        f"{'dir' if directed else 'undir'}_conn.pt"
    )

    # --------------------------------------------------
    # Edge containers
    # --------------------------------------------------
    edges = set()
    deg = np.zeros(n, dtype=np.int32)

    def add_edge(u, v):
        if u == v:
            return
        if max_degree is not None:
            if deg[u] >= max_degree or deg[v] >= max_degree:
                return
        if directed:
            edges.add((u, v))
            deg[u] += 1
        else:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in edges:
                edges.add((a, b))
                deg[a] += 1
                deg[b] += 1

    # ==================================================
    # SKELETON (CONNECT)
    # ==================================================

    # intra-block skeleton (each block connected)
    for b in range(k):
        start = block_start[b]
        size = block_sizes[b]
        if size <= 1:
            continue
        nodes = np.arange(start, start + size, dtype=np.int32)
        rng.shuffle(nodes)
        for i in range(size - 1):
            add_edge(nodes[i], nodes[i + 1])

    # inter-block skeleton (blocks connected together)
    reps = [block_start[b] for b in range(k)]
    for i in range(len(reps) - 1):
        add_edge(reps[i], reps[i + 1])

    # ==================================================
    # SBM EDGE SAMPLING
    # ==================================================

    if mode == 'static':

        for i in range(k):
            u_block = np.arange(
                block_start[i],
                block_start[i] + block_sizes[i],
                dtype=np.int32
            )

            for j in range(i if not directed else 0, k):
                v_block = np.arange(
                    block_start[j],
                    block_start[j] + block_sizes[j],
                    dtype=np.int32
                )

                p = p_in if i == j else p_out
                if p <= 0:
                    continue

                for u in u_block:
                    if max_degree is not None and deg[u] >= max_degree:
                        continue

                    if i == j and not directed:
                        candidates = v_block[v_block > u]
                    else:
                        candidates = v_block

                    if candidates.size == 0:
                        continue

                    # Bernoulli thinning (local, memory-safe)
                    mask = rng.random(candidates.size) < p
                    for v in candidates[mask]:
                        add_edge(u, int(v))

    # ==================================================
    # SAVE
    # ==================================================
    rows = []
    cols = []

    if directed:
        for (u, v) in edges:
            rows.append(u)
            cols.append(v)
    else:
        for (u, v) in edges:
            rows.append(u)
            cols.append(v)
            rows.append(v)
            cols.append(u)

    return save_sbm_graph(rows=rows,cols=cols,n=n,labels=labels,fname=fname,
                          meta=dict(mode=mode,p_in=p_in,p_out=p_out,
                                    block_sizes=block_sizes,max_degree=max_degree),
                                    directed=directed,connected=True)

def generate_temporal_sbm_graph_local(n, k, p_in, p_out, n_steps=10, drift_prob=0.01, directed=False, seed=None,
                                      graph_type='tsbm', change_frac=0.001, enable_add=True, enable_del=True,
                                      max_degree=None, auto_degree_factor=2.0):
    """"
    Params:
        n : int nodes
        k : int communities
        p_in : float probability of edge within community
        p_out : float probability of edge between communities
        n_steps : int number of snaps
        drift_prob : float probability of node's community change
        directed : bool
        seed : int | None
        change_frac: float fraction of E0 that we may change (adds + dels) each step.
        enable_add, enable_del: bool to allow only adds / only dels / both.
        max_degree : int | None hard upper bound on vertex degree
        auto_degree_factor : float multiplier for automatic max_degree
    Returns:
        path (saved .pt)  -- uses save_sbm_graph 
    """
    rng = np.random.default_rng(seed)

    # =====================================================
    # Blocks & labels
    # =====================================================
    base = n // k
    block_sizes = [base] * k
    for i in range(n - base * k):
        block_sizes[i % k] += 1

    block_start = np.cumsum([0] + block_sizes[:-1])
    labels = np.concatenate([
        np.full(sz, i, dtype=np.int32)
        for i, sz in enumerate(block_sizes)
    ])
    labels_per_step = [labels.copy()]

    # =====================================================
    # Automatic max_degree
    # =====================================================
    if max_degree is None:
        exp_deg = []
        for i in range(k):
            ni = block_sizes[i]
            exp_deg.append((ni - 1) * p_in + (n - ni) * p_out)
        max_degree = int(auto_degree_factor * max(exp_deg))
        if max_degree <= 0:
            max_degree = None

    # =====================================================
    # Helpers
    # =====================================================
    def add_edge(edges, deg, u, v):
        if u == v:
            return False
        if max_degree is not None:
            if deg[u] >= max_degree or deg[v] >= max_degree:
                return False

        if directed:
            if (u, v) in edges:
                return False
            edges.add((u, v))
            deg[u] += 1
        else:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edges:
                return False
            edges.add((a, b))
            deg[a] += 1
            deg[b] += 1
        return True

    def remove_edge(edges, deg, e):
        edges.remove(e)
        u, v = e
        deg[u] -= 1
        if not directed:
            deg[v] -= 1

    def edgeset_to_arrays(es):
        if not es:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        arr = np.array(list(es), dtype=np.int64)
        arr = arr[arr[:, 0] != arr[:, 1]]
        order = np.lexsort((arr[:, 1], arr[:, 0]))
        arr = arr[order]
        return arr[:, 0], arr[:, 1]

    # =====================================================
    # t = 0 : SKELETON + SBM
    # =====================================================
    skeleton = set()
    edges = set()
    deg = np.zeros(n, dtype=np.int32)

    # intra-block skeleton
    for b in range(k):
        start = block_start[b]
        size = block_sizes[b]
        if size <= 1:
            continue
        nodes = np.arange(start, start + size)
        rng.shuffle(nodes)
        for i in range(size - 1):
            add_edge(skeleton, deg, nodes[i], nodes[i + 1])

    # inter-block skeleton
    reps = [block_start[b] for b in range(k)]
    for i in range(len(reps) - 1):
        add_edge(skeleton, deg, reps[i], reps[i + 1])

    edges |= skeleton

    # SBM edges
    for i in range(k):
        u_nodes = np.arange(block_start[i], block_start[i] + block_sizes[i])
        for j in range(i if not directed else 0, k):
            v_nodes = np.arange(block_start[j], block_start[j] + block_sizes[j])
            p = p_in if i == j else p_out
            if p <= 0:
                continue

            for u in u_nodes:
                if max_degree is not None and deg[u] >= max_degree:
                    continue
                if i == j and not directed:
                    v_eff = v_nodes[v_nodes > u]
                else:
                    v_eff = v_nodes

                mask = rng.random(v_eff.size) < p
                for v in v_eff[mask]:
                    add_edge(edges, deg, int(u), int(v))

    prev_edges = set(edges)

    rows, cols = edgeset_to_arrays(prev_edges)
    rows_per_step = [rows]
    cols_per_step = [cols]

    # =====================================================
    # Dynamics
    # =====================================================
    E0 = max(1, len(prev_edges))
    eff_E0 = E0 * (2 if not directed else 1)
    num_changes = max(1, int(math.ceil(eff_E0 * change_frac)))
    num_del = num_changes // 2
    num_add = num_changes - num_del

    for t in range(1, n_steps):
        new_edges = set(prev_edges)
        deg = np.zeros(n, dtype=np.int32)
        for (u, v) in new_edges:
            deg[u] += 1
            if not directed:
                deg[v] += 1

        changed = False

        # deletions (non-skeleton only)
        if enable_del:
            removable = list(new_edges - skeleton)
            if removable:
                del_cnt = min(num_del, len(removable))
                to_del = rng.choice(removable, size=del_cnt, replace=False)
                for e in to_del:
                    remove_edge(new_edges, deg, e)
                    changed = True

        #additions
        if enable_add:
            adds = 0
            trials = 0
            max_trials = num_add * 50

            while adds < num_add and trials < max_trials:
                trials += 1
                u = int(rng.integers(0, n))
                v = int(rng.integers(0, n))
                if u == v:
                    continue

                p = p_in if labels[u] == labels[v] else p_out
                if rng.random() < p:
                    if add_edge(new_edges, deg, u, v):
                        adds += 1
                        changed = True

        # CHANGE IN SNAP
        if not changed:
            # try delete
            removable = list(new_edges - skeleton)
            if removable:
                e = removable[rng.integers(len(removable))]
                remove_edge(new_edges, deg, e)
            else:
                # or force one SBM-add
                for _ in range(1000):
                    u = int(rng.integers(0, n))
                    v = int(rng.integers(0, n))
                    if u == v:
                        continue
                    p = p_in if labels[u] == labels[v] else p_out
                    if rng.random() < p:
                        if add_edge(new_edges, deg, u, v):
                            break

        # local drift
        if drift_prob > 0:
            touched = set()
            for (u, v) in new_edges.symmetric_difference(prev_edges):
                touched.add(u)
                touched.add(v)

            for u in touched:
                if rng.random() < drift_prob:
                    old = labels[u]
                    choices = [x for x in range(k) if x != old]
                    if choices:
                        labels[u] = int(rng.choice(choices))

        prev_edges = new_edges
        r, c = edgeset_to_arrays(prev_edges)
        rows_per_step.append(r)
        cols_per_step.append(c)
        labels_per_step.append(labels.copy())

    # =====================================================
    # Save
    # =====================================================
    fname = f"{graph_type}_{n_steps}b_{n}v_{k}c_{'dir' if directed else 'undir'}_conn.pt"
    meta = dict(graph_type=graph_type,p_in=p_in, p_out=p_out, k=k,
                max_degree=max_degree,auto_degree_factor=auto_degree_factor)

    return save_sbm_graph(rows=rows_per_step,cols=cols_per_step,n=n,
                          labels=labels_per_step,fname=fname,meta=meta,
                          temporal=True,directed=directed,connected=True)
  
def test_sbm_generation_and_visualization():
    n = 100          
    k = 4            
    p_in = 0.015      
    p_out = 0.0002

    start = time.time()
    path = generate_sbm_graph_universal(
        n=n, k=k, p_in=p_in, p_out=p_out,
        directed=False, seed = 10, mode = 'static',
        graph_type='sbm'
    )
    print("Generation time:", round(time.time()-start, 3), "sec")
    print(f"Saved in: {path}")

    adj, labels = load_sbm_graph(path)
    print(f"Dim adj: {adj.shape}")
    print(f"Count of edges: {adj._nnz()}")
    print(f"Labels: {torch.unique(labels)}")

    assert adj.shape[0] == n and adj.shape[1] == n, "Dim!"
    assert len(labels) == n, "Len labels!"
    assert adj.is_coalesced(), "Coalesced!"

    A_np = adj.to_dense().cpu().numpy()
    G = nx.from_numpy_array(A_np)
    colors = [plt.cm.tab10(int(c)) for c in labels.cpu().numpy()]

    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42, k=0.15)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=40, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    plt.title("Stochastic Block Model", fontsize=14)
    plt.axis('off')
    plt.savefig("graph.png", dpi=150)
    plt.close()

def test_temporal_sbm_generation():
    # generate
    start = time.time()
    path = generate_temporal_sbm_graph_local(
        n=100,
        k=4,
        p_in=0.15,
        p_out=0.01,
        n_steps=10,
        drift_prob=0.05,
        directed=False,
        seed=42,
        graph_type='tsbm',

        change_frac=0.001,
        enable_add=True,
        enable_del=True
    )
    print("Generation time:", round(time.time() - start, 3), "sec")
    print("Saved in:", path)

    # load
    temporal_adjs, temporal_labels = load_temporal_sbm_graph(path)

    print("Shapes:")
    print("  temporal_adjs:", temporal_adjs.shape)
    print("  temporal_labels:", temporal_labels.shape)

    # basic checks
    assert temporal_adjs.dim() == 3
    assert temporal_adjs.is_sparse
    assert temporal_labels.dim() == 2

    T, n, n2 = temporal_adjs.shape
    assert T == 10
    assert n == 100
    assert n == n2


    # symmetry check for each snapshot (for undirected case)
    print("\nChecking symmetry of snapshots...")
    for t in range(T):
        assert check_snapshot_symmetric(temporal_adjs, t), \
            f"Snapshot {t} is not symmetric"


    # drift sanity
    drift_counts = []
    for t in range(1, T):
        diff = (temporal_labels[t] != temporal_labels[t-1]).sum().item()
        drift_counts.append(diff)
    print("Drift counts per step:", drift_counts)


    # edge-change sanity
    edges_per_step = []
    for t in range(T):
        idx = temporal_adjs[t].coalesce().indices().cpu().numpy()
        edges = set((int(u), int(v)) for u, v in zip(idx[0], idx[1]) if u < v)
        edges_per_step.append(edges)

    base_edges = len(edges_per_step[0])
    print("Edges at t0:", base_edges)

    for t in range(1, T):
        added = len(edges_per_step[t] - edges_per_step[t-1])
        removed = len(edges_per_step[t-1] - edges_per_step[t])

        print(f"t={t}: added={added}, removed={removed}")

        assert added <= base_edges * 0.01 + 5
        assert removed <= base_edges * 0.01 + 5


    # visual
    for t in [0, 3, 6, 9]:
        print(f"Visualizing step {t}...")
        A = temporal_adjs[t].to_dense().cpu().numpy()
        labels = temporal_labels[t].cpu().numpy()

        G = nx.from_numpy_array(A)
        colors = [plt.cm.tab10(l % 10 / 10) for l in labels]

        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        plt.title(f"Step {t}")
        plt.axis("off")
        plt.savefig(f"graph_temp_{t}.png", dpi=150)
        plt.close()
        print(f"Saved graph_temp_{t}.png")

# test_temporal_sbm_generation()
# test_sbm_generation_and_visualization()
