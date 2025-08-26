# External imports
import torch
import os

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
                 subcoms_depth : int = 1):
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
        
        self.adj = adj_matrix

        if features is None:
            self.features = torch.zeros((self.nodes_num,1), dtype=adj_matrix.dtype)
        else:
            self.features = features

        if communities is None:
            n = self.nodes_num
            self.coms = torch.stack((torch.arange(0,n,dtype=torch.long).repeat(n), 
                                     torch.arange(0,n,dtype=torch.long).repeat(n),
                                     torch.arange(0,n,dtype=torch.long).repeat_interleave(n)))
        else:
            self.coms = communities


    def modularity(gamma = 1):
        pass

    def upgrade_graph(self, batch):
        """
        Change the graph based on the current batch of updates.

        Parameters
        ----------
        batch : torch.Tensor of the shape (n,3)
            List of edges with weights given in the form (i,j,w), 
            where i and j are node numbers and w is the changing edge weight. 
        """

        n = self.A.size()[0]
        i, j, w = list(zip(*batch)) # FIXME (to konovalov) use torch hsplit
        new_nodes = set(filter(lambda x: x not in self.index_converter, i+j)) # FIXME (to konovalov) use torch features
        k = len(new_nodes)
        # set indexes (n, n+1, ...) to new nodes
        self.index_converter.update(zip(new_nodes, range(n, n+k)))
        i = [self.index_converter[x] for x in i]
        j = [self.index_converter[x] for x in j]
        update_A = torch.sparse_coo_tensor([i, j], w, (n+k, n+k))
        update_C = torch.sparse_coo_tensor([range(n, n+k), range(n, n+k)], [True for x in range(k)], (n+k, n+k))
        
        # change A, add new lines and columns if necessary
        self.A = add_zero_lines_and_columns(self.A, k, k) + update_A
        # correct D_in and D_out
        self.D_in = torch.cat((self.D_in, torch.zeros(k))) + update_A.sum(dim=0)
        self.D_out = torch.cat((self.D_out, torch.zeros(k))) + update_A.sum(dim=1)
        # correct C by adding new 1-element communities
        self.C = add_zero_lines_and_columns(self.C, k, k) + update_C

        nodes = list(set(i+j)) # affected_nodes
        return torch.sparse_coo_tensor([nodes], torch.ones(len(nodes), dtype=bool), size = (n+k,)) # affected_nodes_mask

    @staticmethod
    def neighborhood(A, nodes, step=1):
        """
        Breadth-First Search method 
        
        Parameters:
            A (torch.sparse_coo): adjacency (n x n).
            nodes (torch.Tensor): binary vector (n,)
            step (int)

        Return:
            torch.Tensor: new binary mask with new nodes.
        """
        # FIXME don't work for sparse A
        visited = nodes.clone()
        A_c = A.coalesce() 

        for k in range(step):
            if not visited.any():
                break

            new_frontier_mask = visited[A_c.indices()[0]]
            neighbors = A_c.indices()[1][new_frontier_mask]
            
            visited[neighbors] = True
            
        return visited
    
    def local_algorithm(self,
                        adj, 
                        features, 
                        method : str, 
                        labels: torch.Tensor | None = None) -> torch.Tensor:
        
        if self.subcoms_depth == 1:
            if method == "magi":
                labels = magi(adj, features, labels)
                res = labels

            elif method == "prgpt":
                res = rough_prgpt(adj.to_sparse(), refine="infomax")

            else:
                raise ValueError("Unsupported baseline method name")
            
            return res.indices()

        else:
            raise ValueError("Unsupported nesting depth of subcommunities")

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
        nodes = torch.nonzero(nodes_mask)

        # Find pairs of all indices (i,l), where i is affected community on level l
        communities = self.coms[0:3:2, torch.isin(self.coms[1], nodes)].unique(dim=1)

        # Find mask of all triples (i,j,l) called affected community triples,  
        # where i is affected community at the level l and j is its node
        mask = (self.coms[0:3:2].unsqueeze(2) == communities.unsqueeze(1))
        affected_mask = torch.sum(mask[0] * mask[1], dim=1, dtype=torch.bool)

        # Find mask of all triples (i,j,Last) called remaining community triples,
        # where i is affected community at the highest level L and j is its node
        remaining_mask = (self.coms[2] == self.subcoms_depth-1) and ~nodes_mask

        # Remove all affected community triples except for the remaining ones
        coms = self.coms[:, ~affected_mask or remaining_mask]

        # Complete the missing community triples from higher levels
        prev_mask = affected_mask and (self.coms[2] == 0)
        for l in range(1, self.subcoms_depth):
            level_mask = affected_mask and (self.coms[2] == l)
            level_coms = self.coms[:2, ~level_mask and prev_mask]
            coms = torch.cat((coms, sparse.ext_range(level_coms, l)), dim=1)
            prev_mask = level_mask

        # Propagate remaining community triples to lower levels and
        # add singleton communities for affected nodes at all levels
        remain_coms = sparse.ext_range(self.coms[:2, remaining_mask], 
                                       self.subcoms_depth-1)
        singleton_coms = sparse.ext_range(torch.tensor([nodes, nodes], dtype = torch.int32), 
                                          self.subcoms_depth)
        coms = torch.cat((coms, remain_coms, singleton_coms), dim=1)

        ext_nodes = self.coms[1,affected_mask].unique()
        ext_nodes_mask = torch.isin(coms[1], ext_nodes)
        self.coms = coms[:, ~ext_nodes_mask]

        for l in range(self.subcoms_depth):
            level_mask = (coms[2] == l)

            # Get affected communites at the level l
            old_coms = coms[:2, ext_nodes_mask and level_mask]

            # Reindexing communities
            reindexing = torch.unique(old_coms[0])
            reindexing_map = { j.item(): i for i, j in enumerate(reindexing) }
            old_coms[0] = torch.tensor([ reindexing_map[i.item()] for i in old_coms[0] ])

            # Aggregate adjacency and features matrices
            reduce_ptn = sparse.tensor(old_coms, (reindexing.size()[0], self.nodes_num),
                                       self.adj.dtype)
            adj = self.aggregate(self.adj, reduce_ptn)
            features = self.aggregate(self.features, reduce_ptn)
            del reduce_ptn

            # Apply local algorithm for aggregated graph
            new_coms = self.local_algorithm(adj, features, "prgpt")

            # Restoring the community of the original graph
            new_coms = torch.tensor([[ reindexing[i.item()] for i in new_coms[0] ],
                                     [ reindexing[i.item()] for i in new_coms[1] ],
                                     torch.full((new_coms.size()[1],), l)], dtype=torch.long)
            res_coms = sparse.mm(new_coms, old_coms, self.size)
            self.coms = torch.cat((self.coms, res_coms), dim=1)


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
