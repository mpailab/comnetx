# External imports
import torch
import os

# Internal imports
from baselines.magi import magi
from baselines.rough_PRGPT import rough_prgpt 
import datasets
from sparse_utils import add_zero_lines_and_columns, to_submatrix, sparse_eye, bool_mm

class Optimizer:
    
    def __init__(self, A : torch.Tensor, X : torch.Tensor = None, C : torch.Tensor = None):

        # matrix of weighted edges - FloatTensor n x n
        self.A = A
        n = self.A.size()[0]

        # Node features
        self.X = X # TODO (to konoval) add features support

        # sum of elements of A in each column  FloatTensor n x 1
        self.D_in = self.A.sum(dim=0).to_dense()
        # sum of elements of A in each line FloatTensor n x 1
        self.D_out = self.A.sum(dim=1).to_dense()

        # communities - BoolTensor k x n
        if C is None:
            C = sparse_eye(n, dtype=bool)
        self.C = C

        self.index_converter = {i:i for i in range(n)} # from old node indexes (in batches) to new (in matrix A)

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

    @staticmethod
    def fast_optimizer_for_small_graph(A, X, method : str, labels: torch.Tensor = None):
        if method == "magi":
            labels = magi(A, X, labels)
            res = labels
        elif method == "prgpt":
            res = rough_prgpt(A.to_sparse(), refine="infomax")
        return res

    @staticmethod
    def aggregation(adj, coms):
        coms = coms.type(adj.dtype)
        tmp = torch.sparse.mm(adj, coms.t())
        return torch.sparse.mm(coms, tmp)

    @staticmethod
    def cut_off(communities : torch.Tensor, nodes: torch.Tensor):
        """
        Return a new community binary matrix by zeroing the node indices of nodes, 
        removing empty communities, and adding single-element communities 
        for each node in nodes.

        Parameters:
            communities (torch.sparse_coo): binary matrix (k x n)
            nodes (torch.Tensor): binary vector (n,)

        Return:
            torch.Tensor: binary matrix (k x n)
        """

        # Reset the node indexes of nodes from communities
        coms = communities * (~ nodes)

        # Remove empty communities # TODO don't work for sparse coms!
        non_zero_coms_mask = coms.any(dim=1)
        filtered_coms = coms[non_zero_coms_mask]

        # Adds single-element communities for each node from nodes
        n = nodes.shape[0]
        resulted_coms = torch.cat((torch.eye(n, n)[nodes], filtered_coms), dim=0).bool()

        return resulted_coms
    
    def run(self, nodes : torch.Tensor):
        """
        Run Optimizer on nodes

        Parameters
        ----------
        nodes : torch.Tensor
        """

        # search affected communities # FIXME don't use to_dense()
        affected_communities_mask = self.C.to_dense()[:, nodes].any(dim=1)
        affected_communities = self.C.to_dense()[affected_communities_mask]
        no_affected_communities = self.C.to_dense()[affected_communities_mask.logical_not()]
        nodes_in_affected_communities = affected_communities.any(dim=0)

        # go to submatrix
        submatrix = to_submatrix(self.A, nodes_in_affected_communities, nodes_in_affected_communities)
        print("submatrix:", submatrix, sep = "\n")
        communities_for_submatrix = affected_communities[:, nodes_in_affected_communities]
        print("communities_for_submatrix:", communities_for_submatrix, sep = "\n")
        nodes_in_submatrix = nodes[nodes_in_affected_communities]
        
        # split affected communities
        splitted_communities_for_submatrix = self.cut_off(communities_for_submatrix, nodes_in_submatrix)
        
        # aggregation     
        aggregated_submatrix = self.aggregation(submatrix, splitted_communities_for_submatrix)

        # fast algorithm for small matrix
        communities_for_aggregated_matrix = self.fast_optimizer_for_small_graph(aggregated_submatrix, None, "prgpt")

        # go to communities for submatrix
        new_communities_for_submatrix = bool_mm(communities_for_aggregated_matrix, splitted_communities_for_submatrix)
        # go to communities for origin matrix
        n = nodes_in_affected_communities.shape[0]
        k = new_communities_for_submatrix.shape[0]
        mask = nodes_in_affected_communities.repeat(k, 1)
        new_communities = torch.zeros(n, dtype = torch.bool).masked_scatter(mask, new_communities_for_submatrix)

        # go to new communities
        self.C = torch.cat((no_affected_communities, new_communities))
    
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
    
# (n1,n2,...,nk)

# v (1,1)
# |
# u (1,1) - w (1,2)