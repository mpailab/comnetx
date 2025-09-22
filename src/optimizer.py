# External imports
import torch
import os
import numpy as np
from typing import Union

# Internal imports
from baselines.magi import magi
from baselines.rough_PRGPT import rough_prgpt
from baselines.leidenalg import leidenalg_partition
import sparse
import datasets
from metrics import Metrics

class Optimizer:
    
    def __init__(self, 
                 adj_matrix : torch.Tensor, 
                 features : torch.Tensor | None = None, 
                 communities : torch.Tensor | None = None,
                 subcoms_num : int = 1,
                 subcoms_depth : int = 1,
                 method : str = "prgpt:infomap"):
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
        self.subcoms_num = subcoms_num
        self.subcoms_depth = subcoms_depth
        
        self.adj = adj_matrix.float() # FIXME add any type support

        if features is None:
            self.features = torch.zeros((self.nodes_num,1), dtype=self.adj.dtype)
        else:
            self.features = features.float() # FIXME add any type support

        if communities is None:
            n = self.nodes_num
            l = self.subcoms_depth
            self.coms = torch.arange(0, n, dtype=torch.long).repeat(l).reshape((l, n))
        else:
            #TODO reindexing of the communities numbers such that the following condition holds:
            # if c is a community number, the node c belongs to the community community c;
            self.coms = communities
        
        self.method = method

    def dense_modularity(self, 
            adj, coms, gamma = 1) -> float:
        """
        Args:
            adj: torch.tensor [n_nodes, n_nodes]
            coms: torch.tensor [n_nodes, n_nodes]
            gamma: float
            
        Returns:
            modularity: float 
        """
        return Metrics.modularity(adj, coms.T, gamma)

    def modularity(self, 
            gamma=1, L=0) -> float:
        """
        Args:
            gamma: float, optional (default=1)
            L: int, optional (default=0)
        Returns:
            modularity: float 
        """
        n = self.coms.shape[1]
        dense_coms = Metrics.create_dense_community(self.coms, n, L).T
        return Metrics.modularity(self.adj, dense_coms.to(torch.float32), gamma)
        

    def update_adj(self,
                      batch: torch.Tensor):
        """
        Change the graph based on the current batch of updates.

        Parameters
        ----------
        batch : torch.Tensor of the shape (n, n)
        """

        batch_size = batch.size()
        if self.size != batch_size:
            raise(f"Unsuitable batch size: {batch_size}. {self.size} is required.")
        
        self.adj += batch.type(self.adj.dtype)
        affected_nodes = batch.indices().unique()

        return affected_nodes

    # @staticmethod
    # def neighborhood_torch(A, nodes, step=1):
    #     """
    #     Args:
    #         A (torch.sparse_coo): adjacency (n x n).
    #         nodes (torch.Tensor): binary vector (n,)
    #         step (int)

    #     Return:
    #         torch.Tensor: new binary mask with new nodes.
    #     """
    #     visited = nodes.clone()
    #     A_c = A.coalesce() 

    #     for k in range(step):
    #         if not visited.any():
    #             break

    #         frontier_mask = visited[A_c.indices()[0]]
    #         neighbors = A_c.indices()[1][frontier_mask]

    #         visited[neighbors] = True
            
    #     return visited
    
    # @staticmethod
    # def neighborhood_sparse(A, nodes, step=1):
    #     """
    #     Args:
    #         A (sparse.COO): adjacency matrix (n x n)
    #         nodes (torch.Tensor): binary vector (n,)
    #         step (int)
        
    #     Return:
    #         torch.Tensor: new binary mask with new nodes
    #     """
    #     visited = nodes.clone().numpy() if isinstance(nodes, torch.Tensor) else nodes.copy()
        
    #     for k in range(step):
    #         if not visited.any():
    #             break
            
    #         # Получаем индексы ненулевых элементов
    #         rows, cols = A.coords
    #         data = A.data
            
    #         # Ищем соседей через индексы
    #         frontier_indices = np.where(visited)[0]
    #         mask = np.isin(rows, frontier_indices)
            
    #         if mask.any():
    #             neighbors = cols[mask]
    #             visited[neighbors] = True
                
    #     return torch.tensor(visited) if isinstance(nodes, torch.Tensor) else visited

    def neighborhood(A: Union[torch.Tensor, 'sparse.COO'], nodes: torch.Tensor, 
                    step: int = 1) -> torch.Tensor:
        """
        Args:
            A : Union[torch.Tensor, sparse.COO]
            nodes : torch.Tensor
            step : int, optional (default=1)
        Returns:
            visited: torch.Tensor 
        """
        visited = nodes.clone()
        
        if isinstance(A, torch.Tensor) and A.is_sparse:
            A_c = A.coalesce()
            for k in range(step):
                if not visited.any():
                    break

                frontier_mask = visited[A_c.indices()[0]]
                neighbors = A_c.indices()[1][frontier_mask]

                alt_frontier_mask = visited[A_c.indices()[1]]
                alt_neighbors = A_c.indices()[0][alt_frontier_mask]
                
                neighbors = torch.cat([neighbors, alt_neighbors])
                neighbors = torch.unique(neighbors)

                visited[neighbors] = True
                
            return visited
                    
        elif hasattr(A, 'coords'):
            rows, cols = A.coords
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
            raise TypeError(f"Unsupported matrix type: {type(A)}")
        
        return visited

    def local_algorithm(self,
                        adj, 
                        features,
                        limited : bool,
                        labels: torch.Tensor | None = None) -> torch.Tensor:
        
        if self.method == "magi":
            return magi(adj, features, labels)

        elif self.method == "prgpt:infomap":
            return rough_prgpt(adj.to_sparse(), refine="infomap")
            
        elif self.method == "prgpt:locale":
            return rough_prgpt(adj.to_sparse(), refine="locale")
        
        elif self.method =="leidenalg":
            return leidenalg_partition(adj.to_sparse())

        else:
            raise ValueError("Unsupported baseline method name")

    @staticmethod
    def aggregate(adj : torch.Tensor, pattern : torch.Tensor):
        return torch.sparse.mm(pattern, torch.sparse.mm(adj, pattern.t()))
    
    
    def run(self, nodes_mask : torch.Tensor):
        """
        Run Optimizer on nodes

        Parameters
        ----------
        nodes_mask : torch.Tensor
        """

        # Find indices of affected nodes
        nodes = torch.nonzero(nodes_mask, as_tuple=True)[0]

        # Find mask of all nodes in the affected communities
        ext_mask = torch.isin(self.coms, self.coms[:, nodes_mask])

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

            # Reindexing communities
            old_idx = torch.unique(coms)
            new_idx = { j.item(): i for i, j in enumerate(old_idx) }
            old_coms = torch.stack((torch.tensor([ new_idx[i.item()] for i in coms ]), 
                                    ext_nodes))
            n = old_idx.size()[0]

            # Aggregate adjacency and features matrices
            aggr_ptn = sparse.tensor(old_coms, (n, self.nodes_num), adj.dtype)
            aggr_adj = self.aggregate(adj, aggr_ptn)
            aggr_features = torch.sparse.mm(aggr_ptn, self.features)
            del aggr_ptn

            # Apply local algorithm for aggregated graph
            coms = self.local_algorithm(aggr_adj, aggr_features, l > 0)

            # Restoring the community of the original graph
            new_coms = torch.stack((torch.tensor([ old_idx[i.item()] for i in coms ]), 
                                    old_idx))
            new_coms = sparse.mm(new_coms, old_coms, self.size)

            # Store new communities at the level l
            self.coms[l, ext_mask[l]] = new_coms[0, new_coms[1].sort().indices]

            # Cut off adjacency matrix
            cut_ptn = sparse.tensor(new_coms, self.size, adj.dtype)
            adj = adj * torch.sparse.mm(cut_ptn.t(), cut_ptn)


    def apply(self, batch):
        """
        Apply Optimizer to the current graph update

        Parameters
        ----------
        batch : torch.Tensor of the shape (n,3)
            List of edges with weights given in the form (i,j,w), 
            where i and j are node numbers and w is the changing edge weight. 
        """

        nodes = self.upgrade_graph(batch)
        nodes = self.neighborhood(self.A, nodes.to_dense()) # BoolTensor n x 1
        self.run(nodes)
