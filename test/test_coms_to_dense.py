import torch

def create_community_matrix(communities, n, L=0):
    communities = communities[L,:].long()
    nodes = torch.tensor(range(n)).long()

    community_matrix = torch.zeros(n, n, dtype=torch.int32)
    community_matrix[communities, nodes] = 1
    
    return community_matrix

communities = torch.tensor([
    [0, 0, 0, 3, 3, 3],
    [0, 0, 2, 3, 4, 3],
    [0, 1, 2, 3, 4, 5]
])

n = communities.shape[1]

result_scatter = create_community_matrix(communities, n, L=1)
print(result_scatter)