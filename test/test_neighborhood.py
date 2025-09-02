import sys
sys.path.append("src") 
from optimizer import Optimizer
import torch
import numpy as np
import sparse
from typing import Union

def neighborhood(A: Union[torch.Tensor, 'sparse.COO'], 
                         nodes: torch.Tensor, 
                         step: int = 1) -> torch.Tensor:
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

A = torch.tensor([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
], dtype=torch.float32)

indices = torch.nonzero(A).t()
values = A[indices[0], indices[1]]

A_torch_sparse = torch.sparse_coo_tensor(
    indices=indices,
    values=values,
    size=A.shape,
    dtype=torch.float32
)

initial_nodes = torch.tensor([False, True, False, False])

nodes_new = neighborhood(A_torch_sparse, initial_nodes, step=1)
print(nodes_new)

torch_sparse = A_torch_sparse.coalesce()
indices = torch_sparse.indices().cpu().numpy()
values = torch_sparse.values().cpu().numpy()
shape = torch_sparse.shape

A_sparse_coo = sparse.COO(
    coords=indices,  
    data=values,     
    shape=shape      
)

result = neighborhood(A_sparse_coo, initial_nodes, step=1)
print(result)