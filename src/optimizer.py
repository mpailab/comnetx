# External imports
import torch
import os

# Internal imports
from baselines.magi import magi
from baselines.rough_PRGPT import rough_prgpt 
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
        
        self.nodes_num = adj_matrix.size()[0]
        self.subcoms_num = subcoms_num
        self.subcoms_depth = subcoms_depth
        
        self.adj = adj_matrix

        if features is None:
            self.features = torch.zeros((self.nodes_num,1), dtype=adj_matrix.dtype)
        else:
            self.features = features

        if communities is None:
            self.coms = torch.eye(self.nodes_num, dtype=torch.int32).repeat(self.subcoms_depth)
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
        nodes = torch.nonzero(nodes_mask)
        coms_nodes_mask = torch.isin(self.coms[1], nodes)

        # Search affected communities
        communities = self.coms[0, coms_nodes_mask].unique()
        communities_mask = torch.isin(self.coms[0], communities)
        coms = self.coms[:, communities_mask]
        ext_coms = self.coms[:, ~communities_mask]
        
        # Split affected communities
        flat_coms = torch.tensor([range(nodes.size()[0]), nodes], dtype = torch.int32)
        for level in range(1, self.subcoms_depth):
            level_mask = coms[2] == level
            level_communities = coms[0, coms_nodes_mask and level_mask].unique()
            mask = ~torch.isin(coms[0], level_communities) and level_mask
            flat_coms = torch.cat((flat_coms, coms[:2, mask]), dim=1)

        # Reindexing communities
        reindexing = torch.unique(flat_coms[0])
        reindexing_map = { j.item(): i for i, j in enumerate(reindexing) }
        flat_coms[0] = torch.tensor([ reindexing_map[i.item()] for i in flat_coms[0] ])

        # Aggregate adjacency and features matrices
        p = torch.sparse_coo_tensor(flat_coms, 
                                    torch.ones(flat_coms.size()[1], dtype=self.adj.dtype),
                                    [reindexing.size()[0], self.nodes_num]).coalesce()
        adj = self.aggregate(self.adj, p)
        features = self.aggregate(self.features, p)

        # Apply local algorithm for aggregated matrices
        coms = self.local_algorithm(adj, features, "prgpt")

        # Restoring the community of the original graph
        self.coms = torch.cat((torch.tensor([[ reindexing[i.item()] for i in coms[0] ],
                                             [ reindexing[i.item()] for i in coms[1] ],
                                             coms[2]]), ext_coms), dim=1)
        
    
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
