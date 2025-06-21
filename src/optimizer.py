import torch

class Optimizer:
    
    def __init__(self, A, C):
        # matrix of weighted edges - FloatTensor n x n
        self.A = A

        # sum of elements of A in each column  FloatTensor n x 1
        self.D_in = self._calculate_D_in()
        # sum of elements of A in each line FloatTensor n x 1
        self.D_out = self._calculate_D_out()

        # communities - BoolTensor k x n 
        self.C = C  # 1, 0, 1

    def _calculate_D_in(self):
        pass

    def _calculate_D_out(self):
        pass

    def modularity(gamma=1):
        pass

    def upgrade_graph(self, batch):
        # change A, add new lines and columns if necessary
        # correct D
        # correct C adding new 1-element communities
        pass

    @staticmethod
    def get_nodes(batch):
        pass

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
    def submatrix(A, lmask, cmask):
        return A[:, cmask][lmask]

    def run(self, batch):
        self.upgrade_graph(batch)
        nodes = self.get_nodes(batch)
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
        self.C = torch.stack((no_affected_communities, new_communities))

if __name__ == "__main__":
    A = torch.tensor([[1,2,3,4], [10,20,30,40], [100,200,300,400], [1000,2000,3000,4000]])
    print(A)
    lmask = torch.BoolTensor([False, True, True, False])
    cmask = torch.BoolTensor([False, True, True, False])
    print(torch.stack((lmask, cmask))) # add line
    B = A[:, cmask][lmask] # submatrix
    print(B)
    print(B.any(dim=1))