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

def generate_temporal_sbm_graph_local(n, k, p_in, p_out, n_steps=10, 
                                      drift_prob=0.01, edge_persistence=0.99,
                                      directed=False, device='cpu', seed=None, ensure_connected=True,
                                      graph_type='tsbm', change_frac=0.001,
                                      enable_add=True, enable_del=True, use_edge_persistence=False,
):
    """
    Temporal SBM generator with controllable local changes.

    Params:
        n : int nodes
        k : int communities
        p_in : float probability of edge within community
        p_out : float probability of edge between communities
        n_steps : int number of snaps
        drift_prob : float probability of node's community change
        directed : bool
        device : str
        seed : int | None
        ensure_connected : bool delete isolated nodes
        change_frac: float fraction of E0 that we may change (adds + dels) each step.
        enable_add, enable_del: bool to allow only adds / only dels / both.
        use_edge_persistence: bool if True -> each previous edge is kept with probability edge_persistence;
                                additions then fill up to change_frac budget.
                                if False -> remove exactly num_del_target 
                                (if enable_del) and add num_add_target (if enable_add).
    Returns:
        path (saved .pt)  -- uses save_sbm_graph 
    """
    rng = np.random.default_rng(seed)

    # block sizes and initial labels
    base_size = n // k
    block_sizes = [base_size] * k
    for i in range(n - base_size * k):
        block_sizes[i % k] += 1
    labels = np.concatenate([np.full(sz, i, dtype=np.int64) for i, sz in enumerate(block_sizes)])
    labels_per_step = [labels.copy()]

    def sample_block_edges(u_nodes, v_nodes, p):
        if len(u_nodes) == 0 or len(v_nodes) == 0 or p <= 0.0:
            return []
        m = len(u_nodes); nv = len(v_nodes)
        total = m * nv
        if total <= 5_000_000:
            mask = rng.random((m, nv)) < p
            ui, vj = np.nonzero(mask)
            return [(int(u_nodes[i]), int(v_nodes[j])) for i, j in zip(ui, vj)]
        else:
            out = []
            for i, u in enumerate(u_nodes):
                mask_row = rng.random(nv) < p
                vj = np.nonzero(mask_row)[0]
                for j in vj:
                    out.append((int(u), int(v_nodes[j])))
            return out

    # initial edges (snapshot 0)
    edges0 = set()
    for i in range(k):
        u_nodes = np.where(labels == i)[0]
        for j in range(i if not directed else 0, k):
            v_nodes = np.where(labels == j)[0]
            if len(u_nodes) == 0 or len(v_nodes) == 0:
                continue
            p = p_in if i == j else p_out
            pairs = sample_block_edges(u_nodes, v_nodes, p)
            if not directed:
                for (a, b) in pairs:
                    if a == b:
                        continue
                    a0, b0 = (a, b) if a < b else (b, a)
                    edges0.add((a0, b0))
            else:
                for (a, b) in pairs:
                    if a == b:
                        continue
                    edges0.add((int(a), int(b)))

    # ensure connectedness cheaply (intra-block)
    if ensure_connected:
        deg = np.zeros(n, dtype=np.int64)
        for (u, v) in edges0:
            deg[u] += 1
            if not directed:
                deg[v] += 1
        isolated = np.where(deg == 0)[0]
        for vtx in isolated:
            block_v = int(labels[vtx])
            same_block = np.where(labels == block_v)[0]
            same_block = same_block[same_block != vtx]
            if len(same_block) > 0:
                u_choice = int(rng.choice(same_block))
                if directed:
                    edges0.add((vtx, u_choice))
                else:
                    a,b = (vtx, u_choice) if vtx < u_choice else (u_choice, vtx)
                    edges0.add((a,b))

    # canonicalize undirected edges (u<v) and remove self-loops / duplicates
    if not directed:
        edges0 = {(u,v) if u < v else (v,u) for (u,v) in edges0 if u != v}
    else:
        edges0 = {(u,v) for (u,v) in edges0 if u != v}

    prev_edges = set(edges0)  # make a copy
    # num_changes determined only by snapshot 0
    E0 = max(1, len(prev_edges))
    # in undirected case treat each undirected edge as 2 units (as requested)
    effective_E0 = E0 * (2 if not directed else 1)
    num_changes = max(1, int(math.ceil(effective_E0 * float(change_frac))))
    # split into add/del targets per step (we will enforce enable_add/enable_del when using)
    num_del_target = num_changes // 2

    rows_per_step = []
    cols_per_step = []

    # helper to convert set of edges -> sorted arrays (unique guaranteed by set)
    def edgeset_to_sorted_arrays(es):
        if not es:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        arr = np.array(list(es), dtype=np.int64)  # shape (E,2)
        # remove possible self-loops just in case
        arr = arr[arr[:,0] != arr[:,1]]
        # sort lexicographically by (u,v)
        order = np.lexsort((arr[:,1], arr[:,0]))
        arr = arr[order]
        return arr[:,0].astype(np.int64), arr[:,1].astype(np.int64)

    # save t=0 (sorted, unique)
    r0, c0 = edgeset_to_sorted_arrays(prev_edges)
    rows_per_step.append(r0); cols_per_step.append(c0)

    # temporal evolution
    for t in range(1, n_steps):
        # start from previous snapshot
        if use_edge_persistence:
            if enable_del:
                # keep edge with probability edge_persistence
                new_edges = {e for e in prev_edges if rng.random() < edge_persistence}
            else:
                new_edges = set(prev_edges)
            deletions_done = len(prev_edges) - len(new_edges)
            remaining_budget = max(0, num_changes - deletions_done)
            add_target = remaining_budget if enable_add else 0
        else:
            if enable_del:
                prev_list = list(prev_edges)
                del_count = min(num_del_target, len(prev_list))
                if del_count > 0:
                    del_idx = rng.choice(len(prev_list), size=del_count, replace=False)
                    to_delete = {prev_list[i] for i in del_idx}
                else:
                    to_delete = set()
                new_edges = set(prev_edges - to_delete)
                deletions_done = len(to_delete)
            else:
                new_edges = set(prev_edges)
                deletions_done = 0
            remaining_budget = max(0, num_changes - deletions_done)

        # additions (try to add up to add_target new edges)
        adds = set()
        while len(adds) < num_changes:
            u = int(rng.integers(0, n)); v = int(rng.integers(0, n))
            if u == v:
                continue
            if not directed:
                a,b = (u,v) if u < v else (v,u)
                if (a,b) in new_edges or (a,b) in adds:
                    continue
                # sample according to SBM probabilities using current labels
                p = p_in if labels[a] == labels[b] else p_out
                if rng.random() < p:
                    adds.add((a,b))
            else:
                if (u,v) in new_edges or (u,v) in adds:
                    continue
                p = p_in if labels[u] == labels[v] else p_out
                if rng.random() < p:
                    adds.add((u,v))
        # apply additions
        new_edges.update(adds)

        # ensure no self-loops, canonicalize undirected and remove duplicates
        if not directed:
            new_edges = {(u,v) if u < v else (v,u) for (u,v) in new_edges if u != v}
        else:
            new_edges = {(u,v) for (u,v) in new_edges if u != v}

        # determine involved nodes (endpoints of adds and deletions) for drift
        involved_nodes = set()
        if use_edge_persistence:
            dels = prev_edges - new_edges
            for (u,v) in dels:
                involved_nodes.add(u); involved_nodes.add(v)
        else:
            if enable_del:
                try:
                    for (u,v) in to_delete:
                        involved_nodes.add(u); involved_nodes.add(v)
                except UnboundLocalError:
                    pass
        for (u,v) in adds:
            involved_nodes.add(u); involved_nodes.add(v)

        # drift for involved nodes
        if len(involved_nodes) > 0 and drift_prob > 0:
            for node in list(involved_nodes):
                if rng.random() < drift_prob:
                    old = int(labels[node])
                    choices = [x for x in range(k) if x != old]
                    if not choices: 
                        continue
                    labels[node] = int(rng.choice(choices))
                    # edges kept as-is (locality)

        # ensure connectedness cheaply
        if ensure_connected:
            deg = np.zeros(n, dtype=np.int64)
            for (u,v) in new_edges:
                deg[u] += 1
                if not directed: deg[v] += 1
            isolated = np.where(deg == 0)[0]
            for vtx in isolated:
                same_block = np.where(labels == labels[vtx])[0]
                same_block = same_block[same_block != vtx]
                if len(same_block) == 0: continue
                u_choice = int(rng.choice(same_block))
                if directed:
                    new_edges.add((vtx, u_choice))
                else:
                    a,b = (vtx, u_choice) if vtx < u_choice else (u_choice, vtx)
                    new_edges.add((a,b))

        # final canonicalization and uniqueness again
        if not directed:
            new_edges = {(u,v) if u < v else (v,u) for (u,v) in new_edges if u != v}
        else:
            new_edges = {(u,v) for (u,v) in new_edges if u != v}

        prev_edges = set(new_edges)

        # convert to sorted arrays and store
        r_arr, c_arr = edgeset_to_sorted_arrays(prev_edges)
        rows_per_step.append(r_arr); cols_per_step.append(c_arr)
        labels_per_step.append(labels.copy())

    # final save
    connected_field = 'conn' if ensure_connected else 'unconn'
    fname = f"{graph_type}_{n_steps}b_{n}v_{k}c_{'dir' if directed else 'undir'}_{connected_field}.pt"
    meta = dict(graph_type=graph_type, n_steps=n_steps, k=k,
                p_in=p_in, p_out=p_out, drift_prob=drift_prob,
                edge_persistence=edge_persistence, change_frac=change_frac,
                enable_add=enable_add, enable_del=enable_del, use_edge_persistence=use_edge_persistence,
                directed=directed, n=n)

    path = save_sbm_graph(
        rows=rows_per_step, cols=cols_per_step, n=n,
        labels=labels_per_step, fname=fname, meta=meta,
        temporal=True, directed=directed, connected=ensure_connected
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
    # generate
    start = time.time()
    path = generate_temporal_sbm_graph_local(
        n=100,
        k=4,
        p_in=0.15,
        p_out=0.01,
        n_steps=10,
        drift_prob=0.05,
        edge_persistence=0.95,
        directed=False,
        seed=42,
        ensure_connected=True,
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

#test_temporal_sbm_generation()

#test_sbm_generation_and_visualization()
