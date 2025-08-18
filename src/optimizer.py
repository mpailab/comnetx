# External imports
import torch
import os

# Internal imports
from baselines.magi import magi
from baselines.rough_PRGPT import rough_prgpt 
import datasets

class Optimizer:
    
    def __init__(self, A, X:torch.Tensor = None, C = None, elements_type = torch.int32):

        # Main type for tensor elements #TODO
        self.elements_type = elements_type

        # matrix of weighted edges - FloatTensor n x n
        self.A = A
        n = self.A.size()[0]

        # matrix of features - FloatTensor n x k
        if X is not None:
            self.X = X

        # Node features
        self.X = None # TODO (to konoval) add features support

        # sum of elements of A in each column  FloatTensor n x 1
        self.D_in = self.A.sum(dim=0).to_dense()
        # sum of elements of A in each line FloatTensor n x 1
        self.D_out = self.A.sum(dim=1).to_dense()

        # communities - BoolTensor k x n
        if C is None:
            # C = torch.eye(n, dtype=bool).to_sparse()
            w = torch.ones(n, dtype=bool)
            i = torch.stack((torch.arange(n), torch.arange(n)))
            C = torch.sparse_coo_tensor(i, w, size=(n, n))
        self.C = C

        self.index_converter = {i:i for i in range(n)} # from old node indexes (in batches) to new (in matrix A)

    def modularity(gamma = 1):
        pass

    @staticmethod
    def add_zero_lines_and_columns(matrix, line_num, col_num):
        zero_columns = torch.sparse_coo_tensor(size=(matrix.size()[0], col_num))
        matrix = torch.cat((matrix, zero_columns), dim=1)
        zero_lines = torch.sparse_coo_tensor(size=(line_num, matrix.size()[1]))
        matrix = torch.cat((matrix, zero_lines), dim=0)
        return matrix

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
        self.A = self.add_zero_lines_and_columns(self.A, k, k) + update_A
        # correct D_in and D_out
        self.D_in = torch.cat((self.D_in, torch.zeros(k))) + update_A.sum(dim=0)
        self.D_out = torch.cat((self.D_out, torch.zeros(k))) + update_A.sum(dim=1)
        # correct C by adding new 1-element communities
        self.C = self.add_zero_lines_and_columns(self.C, k, k) + update_C

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
            res = rough_prgpt(A, labels, refine="infomax")
        return res

    @staticmethod
    def aggregation(adj, coms):
        with coms.float() as x:
            return x.matmul(adj.matmul(x.t()))

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

        # Remove empty communities
        non_zero_coms_mask = (coms.sum(dim=1) != 0)
        filtered_coms = coms[non_zero_coms_mask]

        # Adds single-element communities for each node from nodes.
        n = nodes.shape[0]
        resulted_coms = torch.cat((torch.eye(n, n)[nodes], filtered_coms), dim=0)

        return resulted_coms

    @staticmethod
    def submatrix(matrix, lmask, cmask):
        dense_matrix = matrix.to_dense() if matrix.is_sparse else matrix
        return dense_matrix[:, cmask][lmask]
        #return matrix[:, cmask][lmask]

    def run(self, batch):
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

        # search affected communities
        C_dense = self.C.to_dense()
        affected_communities_mask = C_dense[:, nodes].any(dim=1)
        #affected_communities_mask = self.C[:, nodes].any(dim=1) # BoolTensor n x 1
        #affected_communities = self.C[affected_communities_mask]
        affected_communities = C_dense[affected_communities_mask]
        #no_affected_communities = self.C[affected_communities_mask.logical_not()]
        no_affected_communities = C_dense[affected_communities_mask.logical_not()]
        nodes_in_affected_communities = affected_communities.any(dim=0)

        # go to submatrix
        # affected_communities = self.submatrix(affected_communities, nodes_in_affected_communities, nodes_in_affected_communities)
        submatrix = self.submatrix(self.A, nodes_in_affected_communities, nodes_in_affected_communities)

        # split affected communities
        splitted_communities = self.cut_off(affected_communities, nodes)
        
        # aggregation
        print("submatrix: ", submatrix)
        print("splitted comunities: ", splitted_communities)
        aggregated_submatrix = self.aggregation(submatrix, splitted_communities)

        # fast algorithm for small matrix
        communities_for_aggregated_matrix = self.fast_optimizer_for_small_graph(aggregated_submatrix, None, "prgpt")

        # go to communities for origin matrix
        new_communities = communities_for_aggregated_matrix.matmul(splitted_communities)

        # go to new communities
        self.C = torch.cat((no_affected_communities, new_communities))

if __name__ == "__main__":

    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]])
    #A = torch.tensor([[1,2,3,4], [10,20,30,40], [100,200,300,400], [1000,2000,3000,4000]])
    print(A)
    A = A.to_sparse()
    opt = Optimizer(A)
    batch = [(0, 0, 1), (0, 1, 1), (0, 2, 1), (10, 13, 1)]
    print("batch: ", batch)
    opt.run(batch)


    # opt.neighborhood(opt.A, nodes)
    
    # data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    # dataset = datasets.Dataset("cora", path=data_dir + "/cora")
    # adj, features, labels = dataset.load(tensor_type="coo")
    # opt.fast_optimizer_for_small_graph(adj, features, labels, "magi")


    #opt.run(batch)

# (n1,n2,...,nk)

# v (1,1)
# |
# u (1,1) - w (1,2)