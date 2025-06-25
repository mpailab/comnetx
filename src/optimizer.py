import torch

class Optimizer:
    
    def __init__(self, A, C = None, elements_type = torch.int32):

        # Main type for tensor elements #TODO
        self.elements_type = elements_type

        # matrix of weighted edges - FloatTensor n x n
        self.A = A
        n = self.A.size()[0]

        # sum of elements of A in each column  FloatTensor n x 1
        self.D_in = self.A.sum(dim=0).to_dense()
        # sum of elements of A in each line FloatTensor n x 1
        self.D_out = self.A.sum(dim=1).to_dense()

        # communities - BoolTensor k x n
        if C is None:
            # C = torch.eye(4, dtype=bool).to_sparse()
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
        # batch: [(i, j, w), ...]

        n = self.A.size()[0]
        i, j, w = list(zip(*batch))
        new_nodes = set(filter(lambda x: x not in self.index_converter, i+j))
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
    def neighborhood(nodes, step = 1):
        pass

    @staticmethod
    def fast_optimizer_for_small_graph(A):
        pass

    @staticmethod
    def aggregation(A, C):
        pass

    @staticmethod
    def cut_off(communities, nodes):
        # The result is nonempty communities without nodes + 1-elements communities for each node
        pass

    @staticmethod
    def submatrix(matrix, lmask, cmask):
        return matrix[:, cmask][lmask]

    def run(self, batch):
        nodes = self.upgrade_graph(batch)
        nodes = self.neighborhood(nodes) # BoolTensor n x 1

        # search affected communities
        affected_communities_mask = self.C[:, nodes].any(dim=1) # BoolTensor n x 1
        affected_communities = self.C[affected_communities_mask]
        no_affected_communities = self.C[affected_communities_mask.logical_not()]
        nodes_in_affected_communities = affected_communities.any(dim=0)

        # go to submatrix
        # affected_communities = self.submatrix(affected_communities, nodes_in_affected_communities, nodes_in_affected_communities)
        submatrix = self.submatrix(self.A, nodes_in_affected_communities, nodes_in_affected_communities)

        # split affected communities
        splitted_communities = self.cut_off(affected_communities, nodes)
        
        # aggregation
        aggregated_submatrix = self.aggregation(submatrix, splitted_communities)

        # fast algorithm for small matrix
        new_communities = self.fast_optimizer_for_small_graph(aggregated_submatrix)

        # go to origin matrix
        # new_communities = ... (вернуться к "разжатым сообществам")

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
    nodes = opt.upgrade_graph(batch)
    print("nodes:", nodes.to_dense())