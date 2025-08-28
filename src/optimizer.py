# External imports
import torch
import os
import numpy as np
from typing import Union

# Internal imports
from baselines.magi import magi
from baselines.rough_PRGPT import rough_prgpt 
import sparse
import datasets

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
        communities : torch.Tensor of the shape (3,m)
            Each column in this tensor has the form (i,j,k), where 
                i is a community number, 
                j is a node number that belongs to the community i,
                k is a level of the community i. 
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
            self.coms = torch.stack((torch.arange(0,n,dtype=torch.long).repeat(n), 
                                     torch.arange(0,n,dtype=torch.long).repeat(n),
                                     torch.arange(0,n,dtype=torch.long).repeat_interleave(n)))
        else:
            #TODO reindexing of the communities numbers such that the following condition holds:
            # if c is a community number, the node c belongs to the community community c;
            self.coms = communities
        
        self.method = method


    def modularity(adjacency, assignments, gamma = 1) -> float:
        """
        Args:
            adjacency: SparseTensor or torch.sparse.Tensor or tf.sparse.SparseTensor [n_nodes, n_nodes]
            assignments: torch.Tensor or torch.Tensor or tf.Tensor [n_nodes, n_clusters]
            
        Returns:
            modularity: float 
        """
        if isinstance(adjacency, SparseTensor) and isinstance(assignments, torch.Tensor):
            degrees = adjacency.sum(dim=1)
            m = degrees.sum()
            inv_2m = 1.0 / (2 * m)
            degrees.view(-1, 1)
            a_s = adjacency.matmul(assignments)
            graph_pooled = torch.matmul(a_s.t(), assignments)
            s_d = torch.matmul(assignments.t(), degrees)
            normalizer = torch.matmul(s_d, s_d.t()) * inv_2m
            modularity = (graph_pooled.diag().sum() - normalizer) * inv_2m
            # modularity = torch.trace(graph_pooled - normalizer) * inv_2m
            return modularity.item()
        elif isinstance(adjacency, torch.Tensor) and isinstance(assignments, torch.Tensor):
            degrees = torch.sparse.sum(adjacency, dim=1).to_dense().view(-1, 1)
            m = degrees.sum()
            inv_2m = 1.0 / (2 * m)
            a_s = torch.sparse.mm(adjacency, assignments)
            graph_pooled = torch.matmul(a_s.t(), assignments)
            s_d = torch.matmul(assignments.t(), degrees)
            normalizer = torch.matmul(s_d, s_d.t()) * inv_2m
            modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) * inv_2m
            return modularity.item()
        elif isinstance(adjacency, tf.sparse.SparseTensor) and isinstance(assignments, tf.Tensor):
            degrees = tf.sparse.reduce_sum(adjacency, axis=0)
            m = tf.reduce_sum(degrees)
            inv_2m = 1.0 / (2 * m) 
            degrees = tf.reshape(degrees, (-1, 1))
            a_s = tf.sparse.sparse_dense_matmul(adjacency, assignments)
            graph_pooled = tf.matmul(a_s, assignments, transpose_a=True)
            s_d = tf.matmul(assignments, degrees, transpose_a=True)
            normalizer = tf.matmul(s_d, s_d, transpose_b=True) * inv_2m
            modularity = tf.linalg.trace(graph_pooled - normalizer) * inv_2m
            return modularity.numpy()
        else:
            raise TypeError("Unsupported type")

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

    @staticmethod
    def neighborhood(adj, nodes, step=1):
        """
        Args:
            adj (torch.sparse_coo): adjacency (n x n).
            nodes (torch.Tensor): binary vector (n,)
            step (int)

        Return:
            torch.Tensor: new binary mask with new nodes.
        """
        visited = nodes.clone()
        adj = adj.coalesce() 

        for k in range(step):
            if visited.all():
                break
            new_frontier_mask = visited[adj.indices()[0]]
            neighbors = adj.indices()[1][new_frontier_mask]
            visited[neighbors] = True
            
        return visited
    
    @staticmethod
    def neighborhood_sparse(A, nodes, step=1):
        """
        Args:
            A (sparse.COO): adjacency matrix (n x n)
            nodes (torch.Tensor): binary vector (n,)
            step (int)
        
        Return:
            torch.Tensor: new binary mask with new nodes
        """
        visited = nodes.clone().numpy() if isinstance(nodes, torch.Tensor) else nodes.copy()
        
        for k in range(step):
            if not visited.any():
                break
            
            # Получаем индексы ненулевых элементов
            rows, cols = A.coords
            data = A.data
            
            # Ищем соседей через индексы
            frontier_indices = np.where(visited)[0]
            mask = np.isin(rows, frontier_indices)
            
            if mask.any():
                neighbors = cols[mask]
                visited[neighbors] = True
                
        return torch.tensor(visited) if isinstance(nodes, torch.Tensor) else visited

    @staticmethod
    def neighborhood(A: Union[torch.Tensor, 'sparse.COO'], 
                         nodes: torch.Tensor, 
                         step: int = 1) -> torch.Tensor:
        visited = nodes.clone()
        
        if isinstance(A, torch.Tensor) and A.is_sparse:
            A_c = A.coalesce()
            rows, cols = A_c.indices()
            
            for k in range(step):
                if not visited.any():
                    break

                frontier = torch.where(visited)[0]
                if len(frontier) == 0:
                    break

                mask = torch.isin(rows, frontier)
                neighbors = cols[mask]
                
                if len(neighbors) > 0:
                    visited[neighbors] = True
                    
        elif hasattr(A, 'coords'):
            rows, cols = A.coords
            visited_np = visited.cpu().numpy()
            
            for k in range(step):
                if not visited_np.any():
                    break
                
                frontier_indices = np.where(visited_np)[0]
                mask = np.isin(rows, frontier_indices)
                
                if mask.any():
                    neighbors = cols[mask]
                    visited_np[neighbors] = True
            
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
            labels = magi(adj, features, labels)
            num_nodes = labels.size(0)
            reorder_tensor = torch.stack([labels, torch.arange(num_nodes, device=labels.device)])
            return reorder_tensor

        elif self.method == "prgpt:infomap":
            return rough_prgpt(adj.to_sparse(), refine="infomap")
            
        elif self.method == "prgpt:locale":
            return rough_prgpt(adj.to_sparse(), refine="locale")

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
        nodes = torch.nonzero(nodes_mask).squeeze(1)

        # Find pairs of all indices (i,l), where i is an affected community at the level l
        communities = self.coms[0:3:2, torch.isin(self.coms[1], nodes)].unique(dim=1)

        # Find mask of all triples (i,j,l) called affected community triples,  
        # where i is an affected community at the level l and j is its node
        mask = (self.coms[0:3:2].unsqueeze(2) == communities.unsqueeze(1))
        affected_mask = torch.sum(mask[0] * mask[1], dim=1, dtype=torch.bool)

        # Find nodes in all affected communities
        ext_nodes = self.coms[1,affected_mask].unique()

        # TODO: Perhaps, to save memory, it is worth pruning the matrices self.adj, 
        # self.coms and self.features for nodes from ext_nodes using reindexing

        # Find mask of all triples (i,j,L) called remaining community triples,
        # where i is an affected community at the highest level L and j is its node
        remaining_mask = torch.logical_and(self.coms[2] == self.subcoms_depth-1, ~nodes_mask)

        # Remove all affected community triples except for the remaining ones
        coms = self.coms[:, torch.logical_or(~affected_mask, remaining_mask)]

        # Complete the missing community triples from higher levels
        prev_mask = torch.logical_and(affected_mask, self.coms[2] == 0)
        for l in range(1, self.subcoms_depth):
            level_mask = torch.logical_and(affected_mask, self.coms[2] == l)
            level_coms = self.coms[:2, torch.logical_and(~level_mask, prev_mask)]
            coms = torch.cat((coms, sparse.ext_range(level_coms, l)), dim=1)
            prev_mask = level_mask

        # Propagate remaining community triples to lower levels and
        # add singleton communities for affected nodes at all levels
        remain_coms = sparse.ext_range(self.coms[:2, remaining_mask], 
                                       self.subcoms_depth-1)
        singleton_coms = sparse.ext_range(torch.stack((nodes, nodes)), self.subcoms_depth)
        coms = torch.cat((coms, remain_coms, singleton_coms), dim=1)

        # Reset adjacency matrix
        adj = sparse.reset_matrix(self.adj, ext_nodes)

        # Store communities of unaffected nodes
        ext_nodes_mask = torch.isin(coms[1], ext_nodes)
        self.coms = coms[:, ~ext_nodes_mask]
        coms = coms[:, ext_nodes_mask]

        for l in range(self.subcoms_depth):

            # Get affected communites at the level l
            old_coms = coms[:2, coms[2] == l]

            # Reindexing communities
            reindexing = torch.unique(old_coms[0])
            reindexing_map = { j.item(): i for i, j in enumerate(reindexing) }
            old_coms[0] = torch.tensor([ reindexing_map[i.item()] for i in old_coms[0] ])

            # Aggregate adjacency and features matrices
            aggr_ptn = sparse.tensor(old_coms, (reindexing.size()[0], self.nodes_num), adj.dtype)
            aggr_adj = self.aggregate(adj, aggr_ptn)
            aggr_features = torch.sparse.mm(aggr_ptn, self.features)
            del aggr_ptn

            # Apply local algorithm for aggregated graph
            new_coms = self.local_algorithm(aggr_adj, aggr_features, l > 0)

            # Restoring the community of the original graph
            new_coms = torch.tensor([[ reindexing[i.item()] for i in new_coms[0] ],
                                     [ reindexing[i.item()] for i in new_coms[1] ]])
            res_coms = sparse.mm(new_coms, old_coms, self.size)

            # Store new communities at the level l
            new_coms = torch.cat((new_coms, torch.full((1,new_coms.size()[1]), l)), dim=0)
            self.coms = torch.cat((self.coms, new_coms), dim=1)

            # Cut off adjacency matrix
            cut_ptn = sparse.tensor(res_coms, self.size, adj.dtype)
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
