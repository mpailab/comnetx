# External imports
import torch
from typing import Optional, Callable


# Internal imports
import sparse
from our_utils import print_zone

class Optimizer:

    def __init__(self,
                 adj_matrix: torch.Tensor,
                 features: Optional[torch.Tensor] = None,
                 communities: Optional[torch.Tensor] = None,
                 subcoms_depth: int = 1,
                 method: str = "leidenalg",
                 baseline_iter: int = None,
                 verbose: int = 0,
                 use_gpu: bool = False,
                 aggregation_mode: str = "normalized"):
        """


        Parameters
        ----------
        communities : torch.Tensor of the shape (l,n)
            Each elements communities[d,i] defines a community at the level d
            that the node i belongs to, l is the number of community levels and
            n is the number of nodes.

        aggregation_mode : str
            Pattern aggregation mode. Supported values: "sum", "normalized".
        """
       
        self.size = adj_matrix.size()
        self.nodes_num = adj_matrix.size()[0]            
        self.subcoms_depth = subcoms_depth

        # If GPU mode is requested and CUDA is available, keep all
        # optimizer state on CUDA.
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = adj_matrix.device

        self.adj = adj_matrix.float().to(self.device)

        if features is None:
            self.features = torch.zeros((self.nodes_num, 1), dtype=self.adj.dtype, device=self.device)
            self.synthetic_features = True
            self.has_real_features = False
        else:
            self.features = features.float().to(self.device)
            self.synthetic_features = self._is_sparse_identity(self.features)
            self.has_real_features = not self.synthetic_features

        self.set_communities(communities)
        self.method = method
        self.baseline_iter = baseline_iter
        self.aggregation_mode = self._normalize_aggregation_mode(
            aggregation_mode
        )

        self.verbose = verbose
        self.conversion_time = 0.0
        self.last_timing_info = None
        self.local_algorithm_calls = 0

    @staticmethod
    def _normalize_aggregation_mode(mode: str) -> str:
        mode = mode.lower().strip()
        aliases = {
            "sum": "sum",
            "normalize": "normalized",
            "normalized": "normalized",
        }
        if mode not in aliases:
            supported = ", ".join(["sum", "normalized"])
            raise ValueError(
                f"Unsupported aggregation_mode: {mode}. "
                f"Expected one of: {supported}."
            )
        return aliases[mode]

    @staticmethod
    def _normalized_aggregation_pattern_values(
        counts: torch.Tensor,
        inverse: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        community_weights = counts.to(dtype=dtype).reciprocal()
        return community_weights.index_select(0, inverse)

    @staticmethod
    def _is_sparse_identity(features: torch.Tensor) -> bool:
        if not isinstance(features, torch.Tensor):
            return False
        if features.layout != torch.sparse_coo:
            return False

        feat = features.coalesce()
        if feat.dim() != 2:
            return False

        n, m = feat.shape
        if n != m or feat._nnz() != n:
            return False

        row, col = feat.indices()
        vals = feat.values()

        if not torch.equal(row, col):
            return False
        if not torch.all(vals == 1):
            return False
        if not torch.equal(torch.sort(row).values, torch.arange(n, device=row.device)):
            return False

        return True

    def _aggregate_features(
        self,
        pattern: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if features is None:
            return None

        if isinstance(features, torch.Tensor) and features.layout == torch.sparse_coo:
            features = features.coalesce()

            if self._is_sparse_identity(features):
                return pattern.coalesce()

            return torch.sparse.mm(pattern, features)

        return torch.sparse.mm(pattern, features)

    def _local_algorithm_requires_features(self) -> bool:
        if self.method in {"magi", "dmon"}:
            return True
        if self.method == "dese":
            return self.has_real_features
        if self.method == "s2cag":
            return self.has_real_features
        return False

    def runtime_device(self) -> torch.device:
        return self.device

    def runtime_adj(self) -> torch.Tensor:
        return self.adj

    def runtime_features(self) -> torch.Tensor:
        return self.features
    
    def set_communities(
        self,
        communities: Optional[torch.Tensor],
        replace_subcoms_depth: bool = False,
    ):
        n = self.nodes_num
        l = self.subcoms_depth
        if communities is None:
            self.coms = (
                torch.arange(0, n, dtype=torch.long, device=self.device)
                .repeat(l)
                .reshape((l, n))
            )
        else:
            communities = communities.to(self.device)
            if communities.dim() == 1:
                print(f"Warning: 1D communities converted to 2D with depth 1")
                communities = communities.unsqueeze(0)
            if communities.size(1) != n:
                print(
                    f"Warning: bad communities shape {communities.shape}, "
                    f"required ({l}, {n})"
                )
                print(f"Use default communities with shape ({l}, {n})")
                self.coms = (
                    torch.arange(0, n, dtype=torch.long, device=self.device)
                    .repeat(l)
                    .reshape((l, n))
                )
            else:
                current_depth = communities.size(0)
                if replace_subcoms_depth:
                    self.coms = communities
                    self.subcoms_depth = current_depth
                if current_depth == l:
                    self.coms = communities
                elif current_depth > l:
                    print(
                        f"Warning: communities depth {current_depth} > "
                        f"subcoms_depth {l}."
                    )
                    print(f"Truncating to {l} levels.")
                    self.coms = communities[:l, :]
                else:
                    print(
                        f"Warning: communities depth {current_depth} < "
                        f"subcoms_depth {l}."
                    )
                    print(f"Extending with zeros to {l} levels.")
                    zeros_to_add = torch.zeros(
                        (l - current_depth, n),
                        dtype=communities.dtype,
                        device=self.device,
                    )
                    self.coms = torch.cat([communities, zeros_to_add], dim=0)
   
    def modularity_slow(self,
            gamma: float = 1, L: int = 0, directed: bool = False) -> float:
        """
        Args:
            gamma: float, optional (default=1)
            L: int, optional (default=0)
        Returns:
            modularity: float
        """
        from metrics import Metrics
        return Metrics.modularity_slow(
            self.adj,
            self.coms[L],
            gamma,
            directed=directed,
        )
   
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
        
    def update_adj(
        self,
        batch: torch.Tensor,
        return_mask: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Change the graph based on the current batch of updates.


        Args:
            batch : torch.Tensor of the shape (n, n)
        Returns:


        """

        if self.size != batch.size():
            raise ValueError(
                f"Unsuitable batch size: {batch.size()}. "
                f"{self.size} is required."
            )

        batch_dev = batch if batch.device == self.device else batch.to(self.device)
        if batch_dev.is_sparse and not batch_dev.is_coalesced():
            batch_dev = batch_dev.coalesce()

        self.adj += batch_dev.type(self.adj.dtype)

        if not return_mask:
            return None

        if batch_dev.is_sparse:
            affected_nodes = batch_dev.indices().unique()
        else:
            # For dense updates, track both row and column endpoints as affected nodes.
            nz_rows, nz_cols = torch.nonzero(batch_dev, as_tuple=True)
            affected_nodes = torch.cat((nz_rows, nz_cols), dim=0).unique()
        affected_nodes_mask = torch.zeros(
            self.nodes_num,
            dtype=torch.bool,
            device=self.device,
        )
        affected_nodes_mask[affected_nodes] = True

        return affected_nodes_mask

    @staticmethod
    def neighborhood(adj: torch.Tensor,
                    nodes_mask: torch.Tensor,
                    step: int = 1,
                    is_symmetric=False) -> torch.Tensor:

        visited = nodes_mask.clone()
        if (step <= 0) or visited.all() or not visited.any():
            return visited

        A = adj.coalesce()
        if not is_symmetric:
            idx = A.indices()
            AT = torch.sparse_coo_tensor(idx.flip(0), A.values(), A.shape).coalesce()
            A_sym = (A + AT).coalesce()
        else:
            A_sym = A

        frontier = visited.clone()
        for _ in range(step):
            if not frontier.any() or visited.all():
                break
            y = torch.sparse.mm(
                A_sym,
                frontier.to(dtype=A_sym.dtype).unsqueeze(1),
            ).squeeze(1)
            new_nodes = y > 0
            new_frontier = new_nodes & (~visited)
            visited = visited | new_nodes
            frontier = new_frontier

        return visited

    def local_algorithm(self,
                        adj: torch.Tensor,
                        features: Optional[torch.Tensor],
                        limited: bool = False,
                        labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        timing_info = {'conversion_time' : 0.0}
        self.local_algorithm_calls += 1

        with print_zone(self.verbose >= 3):
            if self.method == "magi":
                from baselines.magi_model import magi
                res = magi(
                    adj,
                    features,
                    labels,
                    n_epochs=self.baseline_iter,
                    timing_info=timing_info,
                )
            elif self.method == "prgpt:infomap":
                from baselines.rough_PRGPT import rough_prgpt
                res = rough_prgpt(adj, refine="infomap", timing_info = timing_info)
            elif self.method == "prgpt:locale":
                from baselines.rough_PRGPT import rough_prgpt
                res = rough_prgpt(adj, refine="locale", timing_info = timing_info)
            elif self.method == "leidenalg":
                from baselines.leiden import leidenalg_partition
                res = leidenalg_partition(adj, timing_info = timing_info)
            elif self.method in ["ldleiden", "dfleiden"]:
                from baselines.dgc import _run_leiden
                res = _run_leiden(self.method, adj, timing_info = timing_info, measure_algorithm_time=True)
            elif self.method == "dmon":
                from baselines.dmon import adapted_dmon
                res = adapted_dmon(
                    adj,
                    features,
                    labels,
                    epochs=self.baseline_iter,
                    timing_info=timing_info,
                )
            elif self.method == "networkit":
                from baselines.network import networkit_partition
                res = networkit_partition(adj, timing_info = timing_info)
            elif self.method == "mfc":
                from baselines.mfc import (
                    mfc_adopted,
                    _binarize_adj,
                    _degree_bins_labels,
                )
                if labels is not None and labels.dim() == 2 and labels.size(0) == 1:
                    labels = labels.squeeze(0)

                return mfc_adopted(
                    adj=adj,
                    labels=labels,
                    network_type="MFC",
                    return_labels=True,
                    timing_info=timing_info,
                    num_epoch=self.baseline_iter,
                )

            elif self.method == "flmig":
                from baselines.flmig import flmig_adopted
                flmig_labels = flmig_adopted(
                    adj=adj,
                    Number_iter=self.baseline_iter,
                    return_labels=True,
                    timing_info=timing_info,
                )
                _, remap = torch.unique(flmig_labels, sorted=True, return_inverse=True)
                res = remap.to(torch.long)
            elif self.method == "dese":
                from baselines.dese import dese
                if not self.has_real_features:
                    raise ValueError("dese can't work without real features")
                res = dese(adj, features, labels, n_epochs=self.baseline_iter, timing_info=timing_info)
            elif self.method == "s2cag":
                from baselines.s2cag import s2cag
                if not self.has_real_features:
                    features = None
                res = s2cag(
                    adj,
                    features,
                    labels,
                    T=self.baseline_iter,
                    timing_info=timing_info,
                )
            else:
                raise ValueError("Unsupported baseline method name")
        self.conversion_time += timing_info.get('conversion_time', 0.0)
        self.last_timing_info = timing_info
        return res

    @staticmethod
    def aggregate(adj: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(pattern, torch.sparse.mm(adj, pattern.t()))

    @staticmethod
    def cut_by_partition(
        adj: torch.Tensor,
        node_mask: torch.Tensor,
        node_labels: torch.Tensor,
        inplace: bool = True,
    ) -> torch.Tensor:
        """
        Cut edges that violate partition constraints.

        If inplace=True (default), zeroes disallowed entries in-place.
        If inplace=False, rebuilds sparse tensor keeping only allowed edges.
        """
        indices = adj.indices()
        row, col = indices

        keep = node_mask[row] & node_mask[col]
        keep = keep & (node_labels[row] == node_labels[col])

        if inplace:
            adj.values().masked_fill_(~keep, 0)
            return adj

        return torch.sparse_coo_tensor(
            indices[:, keep],
            adj.values()[keep],
            adj.size(),
            device=adj.device,
        ).coalesce()
       
    def run(self, nodes_mask: torch.Tensor) -> None:
        """
        Run Optimizer on nodes


        Parameters
        ----------
        nodes_mask : torch.Tensor
        """

        compute_device = self.device
        needs_features = self._local_algorithm_requires_features()

        # Aliases to tensors stored in Optimizer; in-place writes update
        # self.coms directly.
        coms_work = self.coms
        adj_base = self.adj
        features_work = self.features if needs_features else None

        nodes_mask_work = (
            nodes_mask
            if nodes_mask.device == compute_device
            else nodes_mask.to(compute_device)
        )

        # Find indices of affected nodes.
        nodes = torch.nonzero(nodes_mask_work, as_tuple=True)[0]
        if nodes.numel() == 0:
            return

        # Per-level mask: only nodes in communities touched at each level.
        ext_mask_work = torch.zeros_like(coms_work, dtype=torch.bool)
        for l in range(self.subcoms_depth):
            touched = coms_work[l].index_select(0, nodes)
            ext_mask_work[l] = torch.isin(coms_work[l], torch.unique(touched))

        # Set singleton communities for affected nodes at the last level.
        coms_work[-1, nodes_mask_work] = nodes

        # Propagate remaining communities from the larger level to the smaller one.
        for l in range(self.subcoms_depth - 2, -1, -1):
            level_ext_mask = ext_mask_work[l]
            coms_work[l, level_ext_mask] = coms_work[l + 1, level_ext_mask]

        # Reset adjacency matrix to the nodes of affected communities
        affected_nodes_lvl0 = torch.nonzero(ext_mask_work[0], as_tuple=True)[0]
        adj_work = sparse.reset_matrix(adj_base, affected_nodes_lvl0)

        for l in range(self.subcoms_depth):
            # Get affected communites and all their nodes at the level l
            level_ext_mask = ext_mask_work[l]
            coms = coms_work[l, level_ext_mask]
            ext_nodes = torch.nonzero(level_ext_mask, as_tuple=True)[0]

            # Reindex communities
            old_idx, inverse, counts = torch.unique(
                coms,
                sorted=True,
                return_counts=True,
                return_inverse=True,
            )
            n = old_idx.size(0)
            if n == 0:
                continue

            # Aggregate adjacency and features matrices
            aggr_idx = torch.stack((inverse, ext_nodes))
            aggr_adj_ptn = sparse.tensor(
                aggr_idx,
                (n, self.nodes_num),
                adj_work.dtype,
            )
            aggr_adj = self.aggregate(adj_work, aggr_adj_ptn)
            del aggr_adj_ptn

            aggr_features = None
            if needs_features:
                ext_features = features_work.index_select(0, ext_nodes)
                aggr_features = torch.zeros(
                    (n, ext_features.size(1)),
                    dtype=ext_features.dtype,
                    device=ext_features.device,
                )
                aggr_features.index_add_(0, inverse, ext_features)
                aggr_features /= counts.to(dtype=ext_features.dtype).unsqueeze(1)

            # Apply local algorithm for aggregated graph
            coms = self.local_algorithm(aggr_adj, aggr_features, l > 0).to(
                device=compute_device,
                dtype=torch.long,
            )

            # Restoring the community of the original graph
            new_coms = old_idx[coms[inverse]]

            # Restoring the community of the original graph and 
            # store new communities at the level l
            coms_work[l, level_ext_mask] = old_idx[coms[inverse]]

            # Cut off adjacency matrix
            adj_work = self.cut_by_partition(adj_work, level_ext_mask, coms_work[l])
