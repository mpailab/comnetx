import torch

import numpy as np
import os.path
import pickle
import os
import scipy.io as io

import time
import json
import joblib
import pickle
from pathlib import Path

INFO = Path(__file__).parent.parent / "datasets-info" / "json"
PROJECT_DIR = Path(__file__).parent.resolve()

def paths_config_auto_detect():
    machine_default = os.getenv('HOSTNAME')
    machine_parent = os.getenv('PARENT_HOSTNAME') # задать в bash : export PARENT_HOSTNAME=<parent_hostname>
    machine = machine_parent if machine_parent is not None else machine_default
    paths_config_host = f"datasets-info/paths/{machine}.json"
    paths_config_default = "datasets-info/paths/astra.json"
    if os.path.exists(paths_config_host):
        return paths_config_host
    else:
        print(f"Warning! {paths_config_host} does not exist.")
        print(f"         {paths_config_default} will be used instead.")
        return paths_config_default

class Dataset:
    """Dataset treatment"""

    def __init__(self, dataset_name : str, paths_config = None, is_dynamic : bool = True):
        self.name = dataset_name
        self.is_dynamic = is_dynamic
        
        if paths_config is None:
            paths_config  = paths_config_auto_detect()
        with open(paths_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        self.dataset_format = self.detect_dataset_format()
        self.root_section = self.get_root_section(config)
        self.dataset_root = Path(config[self.root_section][self.dataset_format])

        self.adj = None # (l, n, n)-tensor (dynamic) or (n, n)-tensor (static)
        self.is_directed = False
        self.features = None
        self.label = None

    def get_root_section(self, config):
        dynamic = config.get("dynamic", {})
        static = config.get("static", {})
        if self.dataset_format in dynamic:
            return "dynamic"
        elif not self.is_dynamic and self.dataset_format in static:
            return "static"
        else:
            raise KeyError(
                f"Dataset format '{self.dataset_format}' not found in appropriate section. "
                f"is_dynamic={self.is_dynamic}, "
                f"dynamic keys={list(dynamic.keys())}, "
                f"static keys={list(static.keys())}"
            )
        
    def detect_dataset_format(self):
        dname = self.name.lower()
        for file_path in INFO.glob("*.json"):
            fmt = file_path.stem
            with open(file_path) as f:
                if dname in json.load(f):
                    return fmt
        if dname in {"acm", "bat", "citeseer", "cora", "dblp", "eat", "uat"}:
            return "small"
        if dname.startswith("sbm") or dname.startswith("tsbm"):
            return "sbm"
        if dname.startswith("static") or dname.startswith("stream"):
            return "dyn_sbm"
        
        raise ValueError(f"Dataset '{self.name}' not in json config files and no pattern match.")

    def load(self, tensor_type : str = "coo", batches_strategy = None) -> torch.Tensor:
        """
        Load dataset

        Parameters
        ----------
        tensor_type : str
            Type of output tensor: coo, csr, csc, dense
            Default: coo

        Returns
        -------
        adj - (l, n, n)-tensor, where l is number of batches
              (n, n)-tensor for static datasets (batches_strategy is ignored)
        features - ? #TODO
        label - ? #TODO
        """    

        dname = self.name.lower()
        fmt = self.dataset_format

        if not self.is_dynamic and batches_strategy is not None:
            print(f"batches_strategy={batches_strategy} is ignored for static dataset '{self.name}'")

        info_path = INFO / f"{fmt}.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            self.is_directed = (info[dname]["d"] == "directed")
        elif fmt in {"small", "sbm", "dyn_sbm"}:
            self.is_directed = False
        else:
            raise ValueError(f"Unknown dataset_format: {fmt}")

        if self.root_section == "dynamic":
            batches_strategy = "1" if batches_strategy is None else str(batches_strategy)
            if fmt == "konect":
                self._load_konect(batches_strategy=batches_strategy)
            elif fmt == "dyn_attr":
                self._load_dynamic_attr_graph(batches_strategy=batches_strategy)
            elif fmt == "tgc":
                self._load_tgc_graphs(batches_strategy=batches_strategy)
            elif fmt == "dyn_sbm":
                parts = self.name.split("_")
                if dname.startswith("static"):
                    dataset_type, num_snapshots = "static", 5
                elif dname.startswith("stream"):
                    dataset_type, num_snapshots = "stream", 10
                else:
                    raise ValueError(f"Bad dyn_sbm name: {self.name}")
                snap, num_nodes, mu, beta = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
                self._load_prgpt_dataset(dataset_type, num_nodes, mu, beta, snap, num_snapshots)

        if self.root_section == "static":
            if fmt == "magi":
                self._load_npy_format(coo_adj=True)
            elif fmt == "attr":
                self._load_attr_graph()
            elif fmt == "sbm":
                self._load_sbm()
            elif fmt == "ogb":
                raise ValueError(f"Unsupported dataset format: {fmt}")  # FIXME
            else:
                raise ValueError(f"Unsupported dataset format: {fmt}")
        
        # Статический датасет, хранящийся в динамической папке: берём первый срез
        if not self.is_dynamic and self.root_section == "dynamic" and self.adj.dim() == 3:
            if self.adj.shape[0] > 1:
                raise ValueError(
                    f"Static dataset '{self.name}' is stored as dynamic with {self.adj.shape[0]} snapshots. "
                    f"Expected exactly 1 snapshot, but found {self.adj.shape[0]}. "
                )
            self.adj = self.adj[0]

        if tensor_type == "dense":
            self.adj = self.adj.to_dense()
        elif tensor_type == "coo":
            self.adj = self.adj.coalesce()
        elif tensor_type == "csr":
            self.adj = self.adj.to_sparse_csr()
        elif tensor_type == "csc":
            self.adj = self.adj.to_sparse_csc()
        else:
            raise ValueError(f"Unsupported tensor type for torch.sparse: {tensor_type}")
        return self.adj, self.features, self.label

    def _load_npy_format(self, coo_adj=True):
        dname = self.name.lower()
        load_dir = os.path.join(self.dataset_root, dname)

        feat_path = os.path.join(load_dir, f"{dname}_feat.npy")
        label_path = os.path.join(load_dir, f"{dname}_label.npy")
        
        self.features = torch.tensor(np.load(feat_path), dtype=torch.float)
        self.label = torch.tensor(np.load(label_path), dtype=torch.long)
        coo_path = os.path.join(load_dir, f"{dname}_coo_adj.joblib")
        dense_path = os.path.join(load_dir, f"{dname}_adj.npy")
        
        if coo_adj and os.path.exists(coo_path):
            adj_data = joblib.load(coo_path)
            self.adj = torch.sparse_coo_tensor(
                adj_data['indices'], 
                adj_data['values'], 
                size=adj_data['shape']
            )
        elif os.path.exists(dense_path):
            adj_data = np.load(dense_path, allow_pickle=True)
            rows, cols = np.nonzero(adj_data)
            values_np = adj_data[rows, cols]
            
            indices = torch.from_numpy(np.vstack((rows, cols))).long()
            values = torch.from_numpy(values_np).float()
            self.adj = torch.sparse_coo_tensor(indices, values, size=adj_data.shape)
        else:
            raise FileNotFoundError(
                f"No adjacency file found for {dname} in {load_dir}. "
                f"Expected: {coo_path} or {dense_path}"
            )

    def _load_attr_graph(self):
        """Загрузка attributed graphs (flickr, wikics...) из npy/joblib."""
        dname = self.name.lower()
        load_dir = os.path.join(self.dataset_root, dname)
        
        feat_file = os.path.join(load_dir, f"{dname}_feat.npy")
        label_file = os.path.join(load_dir, f"{dname}_label.npy")
        adj_file = os.path.join(load_dir, f"{dname}_coo_adj.joblib")
        
        if not all(os.path.exists(f) for f in [feat_file, label_file, adj_file]):
            raise FileNotFoundError(f"Файлы attr_graph отсутствуют в {load_dir}. Запустите: download.py {dname}")
        
        self.features = torch.tensor(np.load(feat_file), dtype=torch.float)
        self.label = torch.tensor(np.load(label_file), dtype=torch.long)
        adj_data = joblib.load(adj_file)
        self.adj = torch.sparse_coo_tensor(
            adj_data['indices'], adj_data['values'], size=adj_data['shape']
        )

    def _load_dynamic_attr_graph(self, batches_strategy):
        """Загрузка dynamic attr graphs из .npz файлов."""
        dname = self.name.lower()
        dataset_path = os.path.join(self.dataset_root, dname)
        
        pure_name = self.name.split("dyn_")[-1]
        if ":" in batches_strategy:  # p:n стратегия
            p_str, n_str = batches_strategy.split(":")
            p, n = int(p_str), int(n_str)
            sufix = f"{pure_name}-{p+1}_batches.npz"
        else:  # N стратегия
            p, n = 0, int(batches_strategy)
            sufix = f"{pure_name}-{n}_batches.npz"
        dyn_path = os.path.join(dataset_path, f"dynamic_{sufix}")
        feat_path = os.path.join(dataset_path, f"feat_{sufix}")
        
        dyn = np.load(dyn_path, allow_pickle=True)
        i, j = dyn["indices"]
        w, t = dyn["values"], dyn["batch_indices"]
        self.is_directed = bool(dyn.get("is_directed", [False])[0])
        self.adj = self.get_dynamic_adj(i, j, w, t, p, n)

        if feat_path:
            feat_data = np.load(feat_path, allow_pickle=True)
            self.features = torch.from_numpy(feat_data.get("features", None)) if "features" in feat_data else None
            self.label = torch.from_numpy(feat_data.get("labels", None)) if "labels" in feat_data else None

    def _load_tgc_graphs(self, batches_strategy):
        """Загрузка temporal graph clustering (TGC) датасетов из npy/joblib."""
        dname = self.name.lower()
        load_dir = os.path.join(self.dataset_root, dname)
        
        if ":" in batches_strategy:  # p:n стратегия
            p_str, n_str = batches_strategy.split(":")
            p, n = int(p_str), int(n_str)
            adj_file = os.path.join(load_dir, f"{dname}_coo_adj-{p+1}_batches.joblib")
        else:  # N стратегия
            p, n = 0, int(batches_strategy)
            adj_file = os.path.join(load_dir, f"{dname}_coo_adj-{n}_batches.joblib")

        feat_file = os.path.join(load_dir, f"{dname}_feat.npy")
        label_file = os.path.join(load_dir, f"{dname}_label.npy")
        
        if not all(os.path.exists(f) for f in [feat_file, label_file, adj_file]):
            raise FileNotFoundError(f"Для стратегии {batches_strategy} файлы TGC отсутствуют в {load_dir}. Запустите bash: python src/download.py {dname}")
        
        self.features = torch.tensor(np.load(feat_file), dtype=torch.float)
        self.label = torch.tensor(np.load(label_file), dtype=torch.long)
        
        adj_data = joblib.load(adj_file)
        t, i, j = adj_data['indices']
        w = adj_data['values']
        self.adj = self.get_dynamic_adj(i, j, w, t, p, n)

    def _load_prgpt_dataset(self, dataset_type='static',
                       num_nodes=10000,
                       mu=2.5,
                       beta=3.0,
                       snap=1,
                       num_snapshots=5):
        base_path = self.dataset_root
        if dataset_type == 'static':
            edges_path = os.path.join(base_path, f'static_5_{num_nodes}_{mu:.1f}_{beta:.1f}_edges_list.pickle')
            gnd_path = os.path.join(base_path, f'static_5_{num_nodes}_{mu:.1f}_{beta:.1f}_gnd_list.pickle')
        elif dataset_type == 'stream':
            beta_eff = 1.0 if snap == 5 else beta
            edges_path = os.path.join(base_path, f'stream_{snap}_{num_nodes}_{mu:.1f}_{beta_eff:.1f}_edges_list.pickle')
            gnd_path = os.path.join(base_path, f'stream_{snap}_{num_nodes}_{mu:.1f}_{beta_eff:.1f}_gnd.pickle')  
        else:
            raise ValueError("dataset_type must be'static' or 'stream'.")
        
        with open(edges_path, 'rb') as f:
            edges_list = pickle.load(f)
        with open(gnd_path, 'rb') as f:
            if dataset_type == 'static':
                labels_list = pickle.load(f)
            else:
                snap_labels = torch.tensor(pickle.load(f), dtype=torch.long) 

        adjs = []
        all_labels = []
        for snap in range(num_snapshots):
            snap_edges = edges_list[snap]
            if dataset_type == 'static':
                snap_labels = torch.tensor(labels_list[snap], dtype=torch.long)
                all_labels.append(snap_labels) 
            edge_index = torch.tensor(snap_edges, dtype=torch.long).t() 
            values = torch.ones(edge_index.shape[1], dtype=torch.float32)
            adj_sparse = torch.sparse_coo_tensor(indices=edge_index, values=values,
                                                     size=(num_nodes, num_nodes), dtype=torch.float32)

            adj_sparse = adj_sparse.coalesce()
            adj_t = torch.sparse_coo_tensor(indices=adj_sparse.indices().flip(0), 
                                                values=adj_sparse.values(),
                                                size=adj_sparse.shape)
            adj_sparse = (adj_sparse + adj_t).coalesce()
            adjs.append(adj_sparse)


        self.adj = torch.stack(adjs)
        self.label = torch.stack(all_labels) if dataset_type == 'static' else snap_labels
        self.is_directed = False


    def _load_sbm(self, device="cpu"):
        """
        Load sbm dataset
        
        Parameters
        ----------
        data_type : static / temporal
        """
        base_path = self.dataset_root
        file_path = os.path.join(base_path, self.name + ".pt")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"SBM dataset file not found: {file_path}")
        raw = torch.load(file_path, map_location="cpu")
        meta = raw.get("meta", {})
        is_temporal = meta.get("temporal", False)
        is_directed = meta.get("directed", False)

        if is_temporal:
            adjs, labels = self._load_temporal_sbm_graph(file_path, device=device)
        else:
            adjs, labels = self._load_static_sbm_graph(file_path, device=device)

        self.adj = adjs
        self.label = labels
        self.features = None
        self.is_directed = is_directed

    def _load_static_sbm_graph(self, fname, device='cpu'):
        """
        Load .pt file, returns:
        - adj_sparse: torch.sparse_coo_tensor (shape n x n), coalesced, dtype=torch.float32
        - labels_t: torch.tensor(labels, dtype=torch.long)
        """
        if not os.path.isabs(fname) and not os.path.exists(fname):
            candidate = os.path.join(PROJECT_DIR, "test", "graphs", "sbm", fname)
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
    
    def _load_temporal_sbm_graph(self, path, device='cpu'):
        """
        Load dynamic SBM (.pt), returns:
        - adj_sparse: torch.sparse_coo_tensor [n_steps, n, n]
        - labels_t: torch.tensor [n_steps, n]
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
    
    def _load_konect(self, batches_strategy = "1"):
        """
        Загружает граф KONECT в соответствии со стратегией батчинга

        Args:
            batches_strategy: "N" | "p:n" | "real"
                - "N": N равных батчей (готовый файл out.{self.name}.{N}_batches для N = 1, 10, 100, 1000)
                - "p:n" : Стратегия с доминирующим первым батчем.
                      `p` — целое число из ряда 9, 99, 999, ... (соответствует 9%, 99%, 99.9%, ...).
                      Первый батч содержит p% данных, оставшиеся (100-p)% делятся на `n` частей.
                      Пример: "999:10" → 99.9% + 10x0.01% батчей.
                      (файл out.{self.name}.{p+1}_batches + постобработка)
                - "real" : исходные временные метки (файл out.{self.name}.sort)
            По умолчанию "1" (весь граф — один батч).
        """
        # Определение нужного файла
        batches_strategy = str(batches_strategy)
        if batches_strategy == "real":
            filepath = os.path.join(self.dataset_root, self.name, f"out.{self.name}.sort")
        elif ":" in batches_strategy:  # p:n стратегия
            p_str, n_str = batches_strategy.split(":")
            p, n = int(p_str), int(n_str)
            filepath = os.path.join(self.dataset_root, self.name, f"out.{self.name}.{p+1}_batches")
        else:  # N стратегия
            p, n = 0, int(batches_strategy)
            filepath = os.path.join(self.dataset_root, self.name, f"out.{self.name}.{batches_strategy}_batches")

        # Чтение файла
        with open(filepath) as _:
            first_string = _.readline()
            #num_nodes = int(first_string.split()[0])
            edges_num = int(first_string.split()[1])
        i, j, w, t = np.loadtxt(filepath, skiprows=1, dtype=int, unpack=True)
        self.adj = self.get_dynamic_adj(i, j, w, t, p, n)
    
    def get_dynamic_adj(self, i, j, w, t, p, n):
        min_ind = min(i.min(), j.min())
        max_ind = max(i.max(), j.max())
        i -= min_ind
        j -= min_ind
        num_nodes = max_ind - min_ind + 1

        def make_adj(i_arr, j_arr, w_arr):
            idx = np.vstack((i_arr, j_arr))
            adj = torch.sparse_coo_tensor(idx, w_arr, size=(num_nodes, num_nodes)).coalesce()
            return adj if self.is_directed else adj + torch.t(adj)

        if p != 0:
            mask = (t < p)
            adj = make_adj(i[mask], j[mask], w[mask]) # Объединяем первые p батчей в один
            adjs = [adj]

            # Делим последний p-й батч на n равных частей
            mask = (t == p)
            i_last, j_last, w_last = i[mask], j[mask], w[mask]

            step = len(i_last) // n
            remainder = len(i_last) % n
            start = 0
            for part in range(n):
                size = step + (1 if part < remainder else 0)
                end = start + size
                if size > 0:
                    adj = make_adj(i_last[start:end], j_last[start:end], w_last[start:end])
                    adjs.append(adj)
                start = end

        else:
            time_values = np.unique(t)
            masks = [(t == tv) for tv in time_values]
            adjs = [make_adj(i[mask], j[mask], w[mask]) for mask in masks]

        return torch.stack(adjs) # 3-dimensional tensor
    
    def _save_konect(self, path = None, int_weights = True):
        main_adj = self.adj.to_sparse_coo().coalesce()
        edges_num = main_adj._nnz()
        nodes_num = main_adj.size(0)
        if main_adj.ndim == 2:
            batches_strategy = 1
            adjs = [main_adj]
        elif main_adj.ndim == 3:
            batches_strategy = main_adj.shape[0]
            adjs = [main_adj[i] for i in range(batches_strategy)]
        else:
            raise ValueError(f"Unsupported adjacency ndim: {main_adj.ndim}")
        lines = []
        lines_num = edges_num + 1
        lines.append(f"{nodes_num} {lines_num}")
        for num, adj in enumerate(adjs):
            batch = num + 1
            indices = adj.indices()
            values = adj.values()
            for i in range(edges_num):
                source = indices[0, i].item()
                target = indices[1, i].item()
                weight = values[i].item()
                if int_weights:
                    weight = int(weight)
                lines.append(f"{source} {target} {weight} {batch}")
        if path:
            output_dir = os.path.join(path, self.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, f"out.{self.name}.{batches_strategy}_batches")
            with open(filepath, "w") as f:
                for line in lines:
                    f.write(line + '\n')
        else:
            for line in lines:
                print(line)
