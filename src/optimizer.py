# External imports
import torch
from typing import Optional, Callable


# Internal imports
import sparse
from our_utils import print_zone


# Type aliases
LocalAlgorithmFn = Callable[[torch.Tensor, Optional[torch.Tensor], bool, Optional[torch.Tensor]], \
                            torch.Tensor]


class Optimizer:
   
    def __init__(self,
                 adj_matrix: torch.Tensor,
                 features: Optional[torch.Tensor] = None,
                 communities: Optional[torch.Tensor] = None,
                 subcoms_depth: int = 1,
                 method: str = "prgpt:infomap",
                 local_algorithm_fn: Optional[LocalAlgorithmFn] = None,
                 verbose : int = 0,
                 use_gpu: bool = False):
        """


        Parameters
        ----------
        communities : torch.Tensor of the shape (l,n)
            Each elements communities[d,i] defines a community at the level d
            that the node i belongs to, l is the number of community levels and
            n is the number of nodes.
        """
       
        self.size = adj_matrix.size()
        self.nodes_num = adj_matrix.size()[0]
        self.subcoms_depth = subcoms_depth

        # If GPU mode is requested and CUDA is available, keep all optimizer state on CUDA.
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = adj_matrix.device

        self.adj = adj_matrix.float().to(self.device)

        if features is None:
            self.features = torch.zeros((self.nodes_num, 1), dtype=self.adj.dtype, device=self.device)
            self.feat_gen = True
        else:
            self.features = features.float().to(self.device)
            self.feat_gen = False

        self.verbose = verbose
        self.conversion_time = 0.0
        self.last_timing_info = None
        self.local_algorithm_calls = 0
        self.cuda_mem_log = self.verbose >= 4 and self.device.type == "cuda" and torch.cuda.is_available()
        self._log_cuda_memory("init:after-adj-features")

        self.set_communities(communities)
        self._log_cuda_memory("init:after-communities")
        self.method = method
        self.local_algorithm_fn = local_algorithm_fn

    def _local_algorithm_requires_features(self) -> bool:
        if self.local_algorithm_fn is not None:
            return True
        if self.method in {"magi", "dmon", "dese"}:
            return True
        if self.method == "s2cag":
            return not self.feat_gen
        return False

    def _log_cuda_memory(self, stage: str) -> None:
        if not self.cuda_mem_log:
            return
        allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        print(
            f"[cuda-mem] {stage:<22} "
            f"allocated={allocated_mb:9.2f} MB "
            f"reserved={reserved_mb:9.2f} MB "
            f"peak={peak_mb:9.2f} MB"
        )

    @staticmethod
    def _log_cuda_memory_static(stage: str, device: torch.device, enabled: bool) -> None:
        if not enabled:
            return
        allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(
            f"[cuda-mem] {stage:<22} "
            f"allocated={allocated_mb:9.2f} MB "
            f"reserved={reserved_mb:9.2f} MB "
            f"peak={peak_mb:9.2f} MB"
        )

    def runtime_device(self) -> torch.device:
        return self.device

    def runtime_adj(self) -> torch.Tensor:
        return self.adj

    def runtime_features(self) -> torch.Tensor:
        return self.features
    
    def set_communities(self, communities: Optional[torch.Tensor], replace_subcoms_depth: bool = False):
        self._log_cuda_memory("set_communities:start")
        n = self.nodes_num
        l = self.subcoms_depth
        if communities is None:
            self.coms = torch.arange(0, n, dtype=torch.long, device=self.device).repeat(l).reshape((l, n))
        else:
            communities = communities.to(self.device)
            if communities.dim() == 1:
                print(f"Warning: 1D communities converted to 2D with depth 1")
                communities = communities.unsqueeze(0)
            if communities.size(1) != n:
                print(f"Warning: bad communities shape {communities.shape}, required ({l}, {n})")
                print(f"Use default communities with shape ({l}, {n})")
                self.coms = torch.arange(0, n, dtype=torch.long, device=self.device).repeat(l).reshape((l, n))
            else:
                current_depth = communities.size(0)
                if replace_subcoms_depth:
                    self.coms = communities
                    self.subcoms_depth = current_depth
                if current_depth == l:
                    self.coms = communities
                elif current_depth > l:
                    print(f"Warning: communities depth {current_depth} > subcoms_depth {l}.")
                    print(f"Truncating to {l} levels.")
                    self.coms = communities[:l, :]
                else:
                    print(f"Warning: communities depth {current_depth} < subcoms_depth {l}.")
                    print(f"Extending with zeros to {l} levels.")
                    zeros_to_add = torch.zeros((l - current_depth, n), dtype=communities.dtype, device=self.device)
                    self.coms = torch.cat([communities, zeros_to_add], dim=0)
        self._log_cuda_memory("set_communities:end")
   
    def modularity(self,
            gamma: float = 1, L: int = 0, directed: bool = False) -> float:
        """
        Args:
            gamma: float, optional (default=1)
            L: int, optional (default=0)
        Returns:
            modularity: float
        """
        from metrics import Metrics
        return Metrics.modularity(self.adj, self.coms[L], gamma, directed = directed)
        
    def update_adj(self, batch: torch.Tensor, return_mask: bool = True) -> Optional[torch.Tensor]:
        """
        Change the graph based on the current batch of updates.


        Args:
            batch : torch.Tensor of the shape (n, n)
        Returns:


        """

        if self.size != batch.size():
            raise ValueError(f"Unsuitable batch size: {batch.size()}. {self.size} is required.")

        self._log_cuda_memory("update_adj:start")
        batch_dev = batch if batch.device == self.device else batch.to(self.device)
        if batch_dev.is_sparse and not batch_dev.is_coalesced():
            batch_dev = batch_dev.coalesce()
        self._log_cuda_memory("update_adj:prepared")

        self.adj += batch_dev.type(self.adj.dtype)
        self._log_cuda_memory("update_adj:after-add")

        if not return_mask:
            self._log_cuda_memory("update_adj:end-no-mask")
            return None

        if batch_dev.is_sparse:
            affected_nodes = batch_dev.indices().unique()
        else:
            # For dense updates, track both row and column endpoints as affected nodes.
            nz_rows, nz_cols = torch.nonzero(batch_dev, as_tuple=True)
            affected_nodes = torch.cat((nz_rows, nz_cols), dim=0).unique()
        affected_nodes_mask = torch.zeros(self.nodes_num, dtype=torch.bool, device=self.device)
        affected_nodes_mask[affected_nodes] = True
        self._log_cuda_memory("update_adj:end")

        return affected_nodes_mask

    @staticmethod
    def neighborhood(adj: torch.Tensor,
                    nodes_mask: torch.Tensor,
                    step: int = 1,
                    is_symmetric=False,
                    log_cuda: bool = False) -> torch.Tensor:
        cuda_log = log_cuda and adj.device.type == "cuda" and torch.cuda.is_available()
        Optimizer._log_cuda_memory_static("neighborhood:start", adj.device, cuda_log)

        visited = nodes_mask.clone()
        if (step <= 0) or visited.all() or not visited.any():
            Optimizer._log_cuda_memory_static("neighborhood:end-short", adj.device, cuda_log)
            return visited

        A = adj.coalesce()
        if not is_symmetric:
            idx = A.indices()
            AT = torch.sparse_coo_tensor(idx.flip(0), A.values(), A.shape).coalesce()
            A_sym = (A + AT).coalesce()
        else:
            A_sym = A
        Optimizer._log_cuda_memory_static("neighborhood:prepared", adj.device, cuda_log)

        frontier = visited.clone()
        for i in range(step):
            if not frontier.any() or visited.all():
                break
            y = torch.sparse.mm(A_sym, frontier.to(dtype=A_sym.dtype).unsqueeze(1)).squeeze(1)
            new_nodes = y > 0
            new_frontier = new_nodes & (~visited)
            visited = visited | new_nodes
            frontier = new_frontier
            Optimizer._log_cuda_memory_static(f"neighborhood:step-{i}", adj.device, cuda_log)

        Optimizer._log_cuda_memory_static("neighborhood:end", adj.device, cuda_log)
        return visited

    def local_algorithm(self,
                        adj: torch.Tensor,
                        features: Optional[torch.Tensor],
                        limited: bool = False,
                        labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        timing_info = {'conversion_time' : 0.0}
        self.local_algorithm_calls += 1
        self._log_cuda_memory(f"local:start:{self.method}")

        with print_zone(self.verbose >= 3):
            if self.local_algorithm_fn is not None:
                res = self.local_algorithm_fn(adj, features, limited, labels)
            elif self.method == "magi":
                from baselines.magi_model import magi
                res = magi(adj, features, labels, timing_info = timing_info)
            elif self.method == "prgpt:infomap":
                from baselines.rough_PRGPT import rough_prgpt
                res = rough_prgpt(adj, refine="infomap", timing_info = timing_info)
            elif self.method == "prgpt:locale":
                from baselines.rough_PRGPT import rough_prgpt
                res = rough_prgpt(adj, refine="locale", timing_info = timing_info)
            elif self.method == "leidenalg":
                from baselines.leiden import leidenalg_partition
                res = leidenalg_partition(adj, timing_info = timing_info)
            elif self.method == "ldleiden":
                from baselines.ldleiden import ldleiden_partition
                res = ldleiden_partition(adj, timing_info = timing_info)
            elif self.method == "dfleiden":
                from baselines.dfleiden import dfleiden_partition
                res = dfleiden_partition(adj, timing_info = timing_info)
            elif self.method == "dmon":
                from baselines.dmon import adapted_dmon
                res = adapted_dmon(adj, features, labels, timing_info = timing_info)
            elif self.method == "networkit":
                from baselines.network import networkit_partition
                res = networkit_partition(adj, timing_info = timing_info)
            elif self.method == "mfc":
                from baselines.mfc import mfc_adopted, _binarize_adj, _degree_bins_labels
                if labels is not None and labels.dim() == 2 and labels.size(0) == 1:
                    labels = labels.squeeze(0)
                res = mfc_adopted(
                    adj=adj,
                    labels=labels,
                    network_type="MFC",
                    return_labels=True,
                    timing_info=timing_info,
                )
                self._log_cuda_memory("local:end:mfc")
                return res
            elif self.method == "flmig":
                from baselines.flmig import flmig_adopted
                flmig_labels = flmig_adopted(
                    adj=adj,
                    return_labels=True,
                    timing_info=timing_info,
                )
                _, remap = torch.unique(flmig_labels, sorted=True, return_inverse=True)
                res = remap.to(torch.long)
            elif self.method == "dese":
                from baselines.dese import dese
                if self.feat_gen:
                    raise ValueError("dese cann`t work without real features")
                else:
                    res = dese(adj, features, labels, timing_info=timing_info)
            elif self.method == "s2cag":
                from baselines.s2cag import s2cag
                if self.feat_gen:
                    features = None
                res = s2cag(adj, features, labels, timing_info = timing_info)
            else:
                raise ValueError("Unsupported baseline method name")
        self.conversion_time += timing_info.get('conversion_time', 0.0)
        self.last_timing_info = timing_info
        self._log_cuda_memory(f"local:end:{self.method}")
        return res

    @staticmethod
    def aggregate(adj: torch.Tensor, pattern: torch.Tensor, log_cuda: bool = False) -> torch.Tensor:
        cuda_log = log_cuda and adj.device.type == "cuda" and torch.cuda.is_available()
        Optimizer._log_cuda_memory_static("aggregate:start", adj.device, cuda_log)
        aggr_mid = torch.sparse.mm(adj, pattern.t())
        Optimizer._log_cuda_memory_static("aggregate:after-inner", adj.device, cuda_log)
        aggr_out = torch.sparse.mm(pattern, aggr_mid)
        Optimizer._log_cuda_memory_static("aggregate:end", adj.device, cuda_log)
        return aggr_out
       
    def run(self, nodes_mask: torch.Tensor) -> None:
        """
        Run Optimizer on nodes


        Parameters
        ----------
        nodes_mask : torch.Tensor
        """

        compute_device = self.device
        needs_features = self._local_algorithm_requires_features()
        if self.cuda_mem_log:
            torch.cuda.reset_peak_memory_stats(compute_device)
            self._log_cuda_memory("run:start")

        # Aliases to tensors stored in Optimizer; in-place writes update self.coms directly.
        coms_work = self.coms
        adj_base = self.adj
        features_work = self.features if needs_features else None

        nodes_mask_work = nodes_mask if nodes_mask.device == compute_device else nodes_mask.to(compute_device)

        # Find indices of affected nodes.
        nodes = torch.nonzero(nodes_mask_work, as_tuple=True)[0]
        if nodes.numel() == 0:
            self._log_cuda_memory("run:end-empty")
            return

        # Per-level mask: only nodes in communities touched at each level.
        ext_mask_work = torch.zeros_like(coms_work, dtype=torch.bool)
        for l in range(self.subcoms_depth):
            touched = coms_work[l].index_select(0, nodes)
            touched_unique = torch.unique(touched)
            ext_mask_work[l] = torch.isin(coms_work[l], touched_unique)
        self._log_cuda_memory("run:after-ext-mask")

        # Set singleton communities for affected nodes at the last level.
        coms_work[-1, nodes_mask_work] = nodes

        # Propagate remaining communities from the larger level to the smaller one.
        for l in range(self.subcoms_depth - 2, -1, -1):
            level_ext_mask = ext_mask_work[l]
            coms_work[l, level_ext_mask] = coms_work[l + 1, level_ext_mask]

        # Reset adjacency matrix to the nodes of affected communities
        affected_nodes_lvl0 = torch.nonzero(ext_mask_work[0], as_tuple=True)[0]
        adj_work = sparse.reset_matrix(adj_base, affected_nodes_lvl0)
        self._log_cuda_memory("run:after-reset")

        for l in range(self.subcoms_depth):
            self._log_cuda_memory(f"lvl{l}:start")
            # Get affected communites and all their nodes at the level l
            level_ext_mask = ext_mask_work[l]
            coms = coms_work[l, level_ext_mask]
            ext_nodes = torch.nonzero(level_ext_mask, as_tuple=True)[0]

            # Reindex communities
            old_idx, inverse = torch.unique(coms, sorted=True, return_inverse=True)
            n = old_idx.size(0)
            if n == 0:
                continue

            # Aggregate adjacency and features matrices
            aggr_idx = torch.stack((inverse, ext_nodes))
            aggr_ptn = sparse.tensor(aggr_idx, (n, self.nodes_num), adj_work.dtype)
            aggr_adj = self.aggregate(adj_work, aggr_ptn, log_cuda=self.cuda_mem_log)
            aggr_features = (
                torch.sparse.mm(aggr_ptn, features_work)
                if needs_features
                else None
            )
            del aggr_ptn
            self._log_cuda_memory(f"lvl{l}:after-aggregate")

            # Apply local algorithm for aggregated graph
            coms = self.local_algorithm(aggr_adj, aggr_features, l > 0).to(
                device=compute_device,
                dtype=torch.long,
            )
            self._log_cuda_memory(f"lvl{l}:after-local")

            # Restoring the community of the original graph
            new_coms = old_idx[coms[inverse]]
            
            # Store new communities at the level l
            coms_work[l, level_ext_mask] = new_coms


            # Cut off adjacency matrix
            cut_idx = torch.stack((new_coms, ext_nodes))
            cut_ptn = sparse.tensor(cut_idx, self.size, adj_work.dtype)
            cut_mask = torch.sparse.mm(cut_ptn.t(), cut_ptn)
            adj_work = adj_work * cut_mask
            self._log_cuda_memory(f"lvl{l}:after-cut")

        self._log_cuda_memory("run:end")
