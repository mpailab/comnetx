# External imports
import torch
import numpy as np
from typing import Union, Optional, Callable


# Internal imports
import sparse
from metrics import Metrics
from our_utils import print_zone


# Type aliases
LocalAlgorithmFn = Callable[[torch.Tensor, torch.Tensor, bool, Optional[torch.Tensor]], \
                            torch.Tensor]


class Optimizer:
   
    def __init__(self,
                 adj_matrix: torch.Tensor,
                 features: Optional[torch.Tensor] = None,
                 communities: Optional[torch.Tensor] = None,
                 subcoms_depth: int = 1,
                 method: str = "prgpt:infomap",
                 local_algorithm_fn: Optional[LocalAlgorithmFn] = None,
                 verbose : int = 0):
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
       
       
        self.adj = adj_matrix.float() # FIXME add any type support


        if features is None:
            self.features = torch.zeros((self.nodes_num, 1), dtype=self.adj.dtype)
        else:
            self.features = features.float() # FIXME add any type support
       
        self._set_communities(communities)
        self.method = method
        self.local_algorithm_fn = local_algorithm_fn


        self.verbose = verbose
        self.conversion_time = 0.0
    
    def _set_communities(self, communities, replace_subcoms_depth = False):
        n = self.nodes_num
        l = self.subcoms_depth
        if communities is None:
            self.coms = torch.arange(0, n, dtype=torch.long).repeat(l).reshape((l, n))
        else:
            if communities.dim() == 1:
                print(f"Warning: 1D communities converted to 2D with depth 1")
                communities = communities.unsqueeze(0)
            if communities.size(1) != n:
                print(f"Warning: bad communities shape {communities.shape}, required ({l}, {n})")
                print(f"Use default communities with shape ({l}, {n})")
                self.coms = torch.arange(0, n, dtype=torch.long).repeat(l).reshape((l, n))
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
                    zeros_to_add = torch.zeros((l - current_depth, n), dtype=communities.dtype)
                    self.coms = torch.cat([communities, zeros_to_add], dim=0)
   
    def modularity(self,
            gamma: float = 1, L: int = 0, directed: bool = False) -> float:
        """
        Args:
            gamma: float, optional (default=1)
            L: int, optional (default=0)
        Returns:
            modularity: float
        """
        return Metrics.modularity(self.adj, self.coms[L].float(), gamma, directed = directed)
       
    def update_adj(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Change the graph based on the current batch of updates.


        Args:
            batch : torch.Tensor of the shape (n, n)
        Returns:


        """


        if self.size != batch.size():
            raise ValueError(f"Unsuitable batch size: {batch.size()}. {self.size} is required.")
       
        self.adj += batch.type(self.adj.dtype)
        affected_nodes = batch.coalesce().indices().unique()
        affected_nodes_mask = torch.zeros(self.nodes_num, dtype=torch.bool)
        affected_nodes_mask[affected_nodes] = True


        return affected_nodes_mask
   
    @staticmethod
    def neighborhood(adj: Union[torch.Tensor, 'sparse.COO'],
                    nodes_mask: torch.Tensor,
                    step: int = 1) -> torch.Tensor:
        """
        Args:
            adj : Union[torch.Tensor, sparse.COO]
            nodes_mask : torch.BoolTensor
            step : int, optional (default=1)
        Returns:
            visited: torch.BoolTensor
        """
        visited = nodes_mask.clone()


        if step <= 0:
            return visited


        if isinstance(adj, torch.Tensor) and adj.is_sparse:
            # Use sparse matmul for frontier expansion (faster than per-edge masking)
            A = adj.coalesce()
            AT = torch.sparse_coo_tensor(A.indices().flip(0), A.values(), size=A.size()).coalesce()
            A_sym = (A + AT).coalesce()


            # Work with float vector for spmm, keep boolean for masks
            frontier = visited.clone()
            for _ in range(step):
                if not frontier.any():
                    break
                y = torch.sparse.mm(A_sym, frontier.to(dtype=A_sym.dtype).unsqueeze(1)).squeeze(1)
                new_nodes = y > 0
                new_frontier = new_nodes & (~visited)
                visited = visited | new_nodes
                frontier = new_frontier


            return visited
                   
        elif hasattr(adj, 'coords'):
            rows, cols = adj.coords
            visited_np = visited.cpu().numpy()
           
            for k in range(step):
                if not visited_np.any():
                    break
               
                frontier_indices = np.where(visited_np)[0]
               
                mask_out = np.isin(rows, frontier_indices)
                if mask_out.any():
                    neighbors_out = cols[mask_out]
                    visited_np[neighbors_out] = True
               
                mask_in = np.isin(cols, frontier_indices)
                if mask_in.any():
                    neighbors_in = rows[mask_in]
                    visited_np[neighbors_in] = True


            visited = torch.tensor(visited_np, device=visited.device)
           
        else:
            raise TypeError(f"Unsupported matrix type: {type(adj)}")
       
        return visited


    def local_algorithm(self,
                        adj: torch.Tensor,
                        features: torch.Tensor,
                        limited: bool = False,
                        labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        timing_info = {'conversion_time' : 0.0}
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
            elif self.method == "dmon":
                from baselines.dmon import adapted_dmon
                res = adapted_dmon(adj, features, labels, timing_info = timing_info)
            elif self.method == "networkit":
                from baselines.network import networkit_partition
                return networkit_partition(adj, timing_info = timing_info)
            elif self.method == "mfc":
                from baselines.mfc import mfc_adopted, _binarize_adj, _degree_bins_labels
                return mfc_adopted(
                    adj=adj,
                    labels=labels,
                    network_type="MFC",
                    return_labels=True,
                    timing_info=timing_info,
                )
            elif self.method == "flmig":
                from baselines.flmig import flmig_adopted
                labels = flmig_adopted(
                    adj=adj,
                    return_labels=True,
                    timing_info=timing_info,
                )
                uniq, remap = torch.unique(labels, sorted=True, return_inverse=True)
                return remap.to(torch.long)
            elif self.method == "dese":
                from baselines.dese import dese
                res = dese(adj, features, labels, timing_info = timing_info)
            elif self.method == "s2cag":
                from baselines.s2cag import s2cag
                res = s2cag(adj, features, labels, timing_info = timing_info)
            else:
                raise ValueError("Unsupported baseline method name")
        self.conversion_time += timing_info['conversion_time']
        return res

    @staticmethod
    def aggregate(adj: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(pattern, torch.sparse.mm(adj, pattern.t()))
       
    def run(self, nodes_mask: torch.Tensor) -> None:
        """
        Run Optimizer on nodes


        Parameters
        ----------
        nodes_mask : torch.Tensor
        """


        # Find indices of affected nodes
        nodes = torch.nonzero(nodes_mask, as_tuple=True)[0]


        if nodes.numel() == 0:
            return


        # Find mask of all nodes in the affected communities
        # Per-level mask: only nodes in communities touched at each level
        ext_mask = torch.zeros_like(self.coms, dtype=torch.bool)
        for l in range(self.subcoms_depth):
            # Use index_select (faster than boolean indexing)
            touched = self.coms[l].index_select(0, nodes)
            ext_mask[l] = torch.isin(self.coms[l], torch.unique(touched))


        # Set singleton communities for affected nodes at the last level
        self.coms[-1, nodes_mask] = nodes


        # Propagate remaining communities from the larger level to the smaller one
        for l in range(self.subcoms_depth - 2, -1, -1):
            level_ext_mask = ext_mask[l]
            self.coms[l, level_ext_mask] = self.coms[l + 1, level_ext_mask]


        # Reset adjacency matrix to the nodes of affected communities
        adj = sparse.reset_matrix(self.adj, torch.nonzero(ext_mask[0], as_tuple=True)[0])


        for l in range(self.subcoms_depth):


            # Get affected communites and all their nodes at the level l
            coms = self.coms[l, ext_mask[l]]
            ext_nodes = torch.nonzero(ext_mask[l], as_tuple=True)[0]


            # Reindex communities
            old_idx, inverse = torch.unique(coms, sorted=True, return_inverse=True)
            n = old_idx.size(0)


            # Aggregate adjacency and features matrices
            aggr_idx = torch.stack((inverse, ext_nodes))
            aggr_ptn = sparse.tensor(aggr_idx, (n, self.nodes_num), adj.dtype)
            aggr_adj = self.aggregate(adj, aggr_ptn)
            aggr_features = torch.sparse.mm(aggr_ptn, self.features)
            del aggr_ptn


            # Apply local algorithm for aggregated graph
            coms = self.local_algorithm(aggr_adj, aggr_features, l > 0).to(coms.device)


            # Restoring the community of the original graph
            new_coms = old_idx[coms[inverse]]
            
            # Store new communities at the level l
            self.coms[l, ext_mask[l]] = new_coms


            # Cut off adjacency matrix
            cut_idx = torch.stack((new_coms, ext_nodes))
            cut_ptn = sparse.tensor(cut_idx, self.size, adj.dtype)
            adj = adj * torch.sparse.mm(cut_ptn.t(), cut_ptn)