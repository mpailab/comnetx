import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import time
import os


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
    Load dynamic SBM (.pt), returns:
      - adj_sparse: torch.sparse_coo_tensor [n_steps, n, n]
      - labels_t: torch.tensor [n_steps, n]
    """
    data = torch.load(path, map_location=device)
    meta = data.get('meta', {})
    temporal = meta.get('temporal', False)

    if not temporal:
        raise ValueError("File not temporal SBM-graph(temporal=False)")

    n = meta.get('n')
    rows_list = data['rows']
    cols_list = data['cols']
    labels_list = data['labels']
    n_steps = len(rows_list)

    adjs = []
    for rows, cols in zip(rows_list, cols_list):

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        if len(rows) == 0:
            indices = torch.empty((2, 0), dtype=torch.long, device=device)
            values = torch.empty((0,), dtype=torch.float32, device=device)
        else:
            idx_np = np.vstack([rows, cols])          # shape (2, E)
            indices = torch.from_numpy(idx_np).long().to(device)
            values = torch.ones(len(rows), dtype=torch.float32, device=device)
        adj_t = torch.sparse_coo_tensor(indices, values, (n, n), device=device).coalesce()
        adjs.append(adj_t)
    temporal_adjs = torch.stack(adjs, dim=0)
    temporal_labels = torch.stack(labels_list, dim=0).to(device)

    return temporal_adjs, temporal_labels

def generate_sbm_graph_universal(n, k, p_in, p_out, 
                       batch_size=None, directed=False, device='cpu', seed=None, 
                       ensure_connected=True, mode='auto', graph_type='sbm'
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
    Returns:
        if mode='static'  (adj, labels)
        if mode='batch'  (adj_batches, label_batches)
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
    # labels_t = torch.tensor(labels, dtype=torch.long, device=device) 

    if mode == 'auto':
        if n >= 10000 or (batch_size and batch_size < n):
            mode = 'batch'
        else:
            mode = 'static'

    batches_field = batch_size if (mode == 'batch' and batch_size is not None) else 0
    connected_field = 'conn' if ensure_connected else 'unconn'
    fname = (
        f"{graph_type}_{batches_field}b_{n}v_{k}c_"
        f"{'dir' if directed else 'undir'}_"
        f"{connected_field}.pt"
    )
    if mode == 'static':
        rows = []
        cols = []
        for i in range(k):
            si = block_start[i]
            ni = block_sizes[i]
            u = np.arange(si, si + ni, dtype=np.int64)
            j_range = range(i, k) if not directed else range(0, k)
            for j in j_range:
                sj = block_start[j]
                nj = block_sizes[j]
                v = np.arange(sj, sj + nj, dtype=np.int64)

                p = p_in if i == j else p_out

                if ni == 0 or nj == 0:
                    continue

                if not directed and i == j:
                    mask = rng.random((ni, ni)) < p
                    ui, vj = np.nonzero(np.triu(mask, k=1))
                    if ui.size > 0:
                        rows.extend(u[ui].tolist())
                        cols.extend(v[vj].tolist())
                        rows.extend(v[vj].tolist())
                        cols.extend(u[ui].tolist())

                else:
                    mask = rng.random((ni, nj)) < p
                    ui, vj = np.nonzero(mask)
                    if ui.size > 0:
                        rows.extend((u[ui]).tolist())
                        cols.extend((v[vj]).tolist())
                        if (not directed) and (j != i):
                            rows.extend((v[vj]).tolist())
                            cols.extend((u[ui]).tolist())

        if ensure_connected:
            if len(rows) == 0:
                for b in range(k):
                    if block_sizes[b] >= 2:
                        a = block_start[b]
                        bidx = block_start[b] + 1
                        rows.append(a); cols.append(bidx)
                        if not directed:
                            rows.append(bidx); cols.append(a)
            deg = np.zeros(n, dtype=np.int64)
            if len(rows) > 0:
                np.add.at(deg, np.asarray(rows, dtype=np.int64), 1)
                np.add.at(deg, np.asarray(cols, dtype=np.int64), 1)
            isolated = np.where(deg == 0)[0]
            for vtx in isolated:
                block_v = labels[vtx]
                same_block_start = block_start[block_v]
                same_block_size = block_sizes[block_v]
                if same_block_size <= 1:
                    continue
                candidates = np.arange(same_block_start, same_block_start + same_block_size, dtype=np.int64)
                candidates = candidates[candidates != vtx]
                u_choice = int(rng.choice(candidates))
                rows.append(int(vtx)); cols.append(u_choice)
                if not directed:
                    rows.append(int(u_choice)); cols.append(int(vtx))
        saved_path = save_sbm_graph(rows=rows, cols=cols, n=n, labels=labels, fname=fname,
                                    meta={"mode": "static", "p_in": p_in, "p_out": p_out, "block_sizes": block_sizes},
                                    directed=directed, connected=ensure_connected)
        return saved_path

    # --- BATCH mode
    elif mode == 'batch':
        if batch_size is None:
            batch_size = 10

        rows = []
        cols = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_labels = labels[start:end]
            batch_n = end - start
            for i in range(k):
                local_idx = np.where(batch_labels == i)[0]
                if local_idx.size == 0:
                    continue
                u_local_global = (start + local_idx).astype(np.int64) 

                for j in range(k):
                    sj = block_start[j]
                    nj = block_sizes[j]
                    if nj == 0:
                        continue
                    v_global = np.arange(sj, sj + nj, dtype=np.int64)

                    p = p_in if i == j else p_out

                    if not directed and i == j:
                        mask = rng.random((u_local_global.size, v_global.size)) < p
                        ui, vj = np.nonzero(mask)
                        if ui.size > 0:
                            gu = u_local_global[ui]
                            gv = v_global[vj]
                            sel = np.where(gu < gv)[0]
                            if sel.size > 0:
                                rows.extend(gu[sel].tolist())
                                cols.extend(gv[sel].tolist())
                                rows.extend(gv[sel].tolist())
                                cols.extend(gu[sel].tolist())
                    else:
                        mask = rng.random((u_local_global.size, v_global.size)) < p
                        ui, vj = np.nonzero(mask)
                        if ui.size > 0:
                            rows.extend(u_local_global[ui].tolist())
                            cols.extend(v_global[vj].tolist())
                            if not directed and i != j:
                                rows.extend(v_global[vj].tolist())
                                cols.extend(u_local_global[ui].tolist())

        if ensure_connected:
            if len(rows) == 0:
                for b in range(k):
                    if block_sizes[b] >= 2:
                        a = block_start[b]
                        bidx = block_start[b] + 1
                        rows.append(a); cols.append(bidx)
                        if not directed:
                            rows.append(bidx); cols.append(a)

            deg = np.zeros(n, dtype=np.int64)
            if len(rows) > 0:
                np.add.at(deg, np.asarray(rows, dtype=np.int64), 1)
                np.add.at(deg, np.asarray(cols, dtype=np.int64), 1)
            isolated = np.where(deg == 0)[0]
            for vtx in isolated:
                block_v = labels[vtx]
                same_block_start = block_start[block_v]
                same_block_size = block_sizes[block_v]
                if same_block_size <= 1:
                    continue
                candidates = np.arange(same_block_start, same_block_start + same_block_size, dtype=np.int64)
                candidates = candidates[candidates != vtx]
                u_choice = int(rng.choice(candidates))
                rows.append(int(vtx)); cols.append(u_choice)
                if not directed:
                    rows.append(int(u_choice)); cols.append(int(vtx))

        saved_path = save_sbm_graph(rows=rows, cols=cols, n=n, labels=labels, fname=fname,
                                    meta={"mode": "batch", "batch_size": batch_size, "p_in": p_in, "p_out": p_out, "block_sizes": block_sizes},
                                    directed=directed, connected=ensure_connected)
        return saved_path

    else:
        raise ValueError(f"Error mode: {mode}")
    

def generate_temporal_sbm_graph_optimized(
    n, k, p_in, p_out, n_steps=10, drift_prob=0.01, edge_persistence=0.9,
    directed=False, device='cpu', seed=None, ensure_connected=True,
    graph_type='tsbm'
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
    Returns:
        temporal_adjs: 3d tensor [n_steps, n, n]
        temporal_labels: 2d tensor [n_steps, n]
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
    def gen_edges(labels):
        rows, cols = [], []
        for i in range(k):
            u = np.where(labels == i)[0]
            for j in range(k):
                v = np.where(labels == j)[0]
                if len(u) == 0 or len(v) == 0:
                    continue
                p = p_in if i == j else p_out
                mask = rng.random((len(u), len(v))) < p
                ui, vj = np.nonzero(mask)
                rows.extend(u[ui])
                cols.extend(v[vj])
        rows = np.array(rows, dtype=np.int64)
        cols = np.array(cols, dtype=np.int64)

        # Remove self-loops
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]

        if not directed:
            # Mirror edges and remove duplicates
            mirrored = np.vstack([cols, rows])
            combined = np.hstack([np.vstack([rows, cols]), mirrored])
            combined = np.unique(combined, axis=1)
            rows, cols = combined[0], combined[1]

        return rows, cols

    # --- initial snapshot
    rows, cols = gen_edges(labels)
    edges_per_step = [(rows, cols)]
    labels_per_step = [labels.copy()]
    prev_edges = set(zip(rows.tolist(), cols.tolist()))

    for t in range(1, n_steps):
        # --- node drift
        drift_mask = rng.random(n) < drift_prob
        for i in np.where(drift_mask)[0]:
            old = labels[i]
            labels[i] = rng.choice([x for x in range(k) if x != old])

        # --- edge persistence
        new_edges = set()
        for u, v in prev_edges:
            if rng.random() < edge_persistence:
                new_edges.add((u, v))

        # --- new edges according to SBM
        rows_new, cols_new = gen_edges(labels)
        for u, v in zip(rows_new, cols_new):
            if rng.random() < (1 - edge_persistence):
                new_edges.add((u, v))

        # --- ensure connectedness
        if ensure_connected:
            deg = np.zeros(n, dtype=np.int64)
            for u, v in new_edges:
                deg[u] += 1
                deg[v] += 1
            isolated = np.where(deg == 0)[0]
            for vtx in isolated:
                same_block = np.where(labels == labels[vtx])[0]
                same_block = same_block[same_block != vtx]
                if len(same_block) > 0:
                    u_choice = int(rng.choice(same_block))
                    new_edges.add((vtx, u_choice))
                    if not directed:
                        new_edges.add((u_choice, vtx))

        # --- mirror undirected edges
        if not directed:
            mirrored = {(v, u) for u, v in new_edges}
            new_edges.update(mirrored)

        # --- finalize snapshot
        prev_edges = new_edges
        if new_edges:
            r, c = zip(*new_edges)
            r = np.array(r, dtype=np.int64)
            c = np.array(c, dtype=np.int64)
        else:
            r = np.array([], dtype=np.int64)
            c = np.array([], dtype=np.int64)

        edges_per_step.append((r, c))
        labels_per_step.append(labels.copy())


    connected_field = 'conn' if ensure_connected else 'unconn'
    fname = f"{graph_type}_{n_steps}s_{n}v_{k}c_{'dir' if directed else 'undir'}_{connected_field}.pt"
    meta = dict(
        graph_type=graph_type, n_steps=n_steps, k=k,
        p_in=p_in, p_out=p_out, drift_prob=drift_prob,
        edge_persistence=edge_persistence
    )

    path = save_sbm_graph(
        rows=[r for r, _ in edges_per_step],
        cols=[c for _, c in edges_per_step],
        n=n,
        labels=labels_per_step,
        fname=fname,
        meta=meta,
        temporal=True,
        directed=directed,
        connected=ensure_connected
    )
    return path


def test_sbm_generation_and_visualization():
    n = 100          
    k = 4            
    p_in = 0.015      
    p_out = 0.0002     

    start = time.time()
    path = generate_sbm_graph_universal(
        n=n, k=k, p_in=p_in, p_out=p_out,
        directed=False, seed = 42,
        ensure_connected=True, mode = 'static',
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

    # A_np = adj.to_dense().cpu().numpy()
    # G = nx.from_numpy_array(A_np)
    # colors = [plt.cm.tab10(int(c)) for c in labels.cpu().numpy()]

    # plt.figure(figsize=(7, 7))
    # pos = nx.spring_layout(G, seed=42, k=0.15)
    # nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=40, alpha=0.8)
    # nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    # plt.title("Stochastic Block Model", fontsize=14)
    # plt.axis('off')
    # plt.savefig("graph.png", dpi=150)
    # plt.close()

def test_temporal_sbm_generation():
    start = time.time()
    path = generate_temporal_sbm_graph_optimized(
        n=100,        
        k=4,
        p_in=0.15,
        p_out=0.01,
        n_steps=10,
        drift_prob=0.05,
        edge_persistence=0.9,
        directed=False,
        seed=42,
        ensure_connected=True,
        graph_type='tsbm'
    )
    print("Generation time for temporal:", round(time.time()-start, 3), "sec")
    print(f"Saved in: {path}")

    temporal_adjs, temporal_labels = load_temporal_sbm_graph(path)
    print("Dim:")
    print("  temporal_adjs:", temporal_adjs.shape, "(sparse)")
    print("  temporal_labels:", temporal_labels.shape)

    n_steps = temporal_labels.shape[0]
    n = temporal_labels.shape[1]
    assert n_steps == 10, "Count of steps!"
    assert n == 100, "Count of nodes!"

    for t in [0, 3, 6, 9]:
        print(f"Visualizing step {t}...")
        A_t = temporal_adjs[t].to_dense().cpu().numpy()
        labels_t = temporal_labels[t].cpu().numpy()

        G = nx.from_numpy_array(A_t)
        colors = [plt.cm.tab10(l % 10 / 10) for l in labels_t]

        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        plt.title(f"Step {t}")
        plt.axis("off")
        plt.savefig(f"graph_temp_{t}.png", dpi=150)
        plt.close()
        print(f"Saved graph_temp_{t}.png")





#test_sbm_generation_and_visualization()
test_temporal_sbm_generation()
