import torch

# def create_community_matrix(communities, n_communities, n_nodes):
#     # n_communities = int(communities[0].max())
#     # n_nodes = int(communities[1].max()) + 1
    
#     # Создаем индексы
#     comms = (communities[0]).long()  # индексы сообществ (0-based)
#     nodes = communities[1].long()        # индексы узлов
    
#     # Создаем матрицу: строки - сообщества, столбцы - узлы
#     community_matrix = torch.zeros(n_communities, n_nodes, dtype=torch.int32)
    
#     # Заполняем матрицу с помощью индексации
#     community_matrix[comms, nodes] = 1
    
#     return community_matrix

def create_community_matrix(communities, n, L=0):
   
    communities = communities[0:2,L*n:(L+1)*n]

    comms = (communities[0]).long()
    nodes = communities[1].long()
    
    community_matrix = torch.zeros(n, n, dtype=torch.int32)
    community_matrix[comms, nodes] = 1
    
    return community_matrix

communities = torch.tensor([
    [0, 0, 0, 3, 3, 3, 0, 0, 2, 3, 4, 3, 0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
])

result_scatter = create_community_matrix(communities, n=6, L=2)
print(result_scatter)