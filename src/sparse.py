import torch
from typing import Any


def tensor(indices : torch.Tensor, size : torch.types._size, dtype : torch.dtype):
    return torch.sparse_coo_tensor(indices, 
                                   torch.ones(indices.size()[1], dtype=dtype),
                                   size).coalesce()


def mm(indices1 : torch.Tensor, indices2 : torch.Tensor, size : torch.Size):
    # FIXME add bool type support
    return torch.sparse.mm(tensor(indices1, size, torch.float), 
                           tensor(indices2, size, torch.float)).indices()


def ext_range(tensor : torch.Tensor, size : int):
    A = tensor.repeat(1,size)
    B = torch.arange(0,size).repeat_interleave(tensor.size()[1]).unsqueeze(0)
    return torch.cat((A, B), dim=0)


def reset_matrix(tensor : torch.Tensor, 
                 indices : torch.Tensor) -> torch.Tensor:
    mask = torch.isin(tensor.coalesce().indices(), indices).all(0)
    return torch.sparse_coo_tensor(tensor.coalesce().indices()[:, mask], 
                                   tensor.coalesce().values()[mask],
                                   tensor.size()).coalesce()


def reset(tensor : torch.Tensor, 
          key : int | slice | torch.Tensor,
          dim : int = 0,
          invert : bool = False) -> torch.Tensor:
    """
    Reset to zero values of a sparse tensor in COO format by key in a given dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Sparse tensor in COO format
    key : int | slice | torch.Tensor
        Key for index selection
    dim : int
        Dimension
        Default: 0
    invert : bool
        Reset to zero all values whose indexes do not match the key
    
    Returns
    -------
        torch.Tensor
    """
    selected_indices = torch.arange(tensor.size()[dim])[key]
    mask = torch.isin(tensor.coalesce().indices()[dim], selected_indices) ^ invert
    return torch.sparse_coo_tensor(tensor.coalesce().indices()[:, mask], 
                                   tensor.coalesce().values()[mask],
                                   tensor.size()).coalesce()


def clear(tensor : torch.Tensor,
          dim : int = 0) -> torch.Tensor:
    """
    Remove zero subtensors from sparse tensor in COO format in a given dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Sparse tensor in COO format
    dim : int
        Dimension
        Default: 0
    
    Returns
    -------
        torch.Tensor
    """
    indices = tensor.coalesce().indices()
    size = list(tensor.size())

    old_dim_indices = indices[dim]
    unique_old_dim_indices = torch.unique(old_dim_indices)
    new_dim_map = {
        old_val.item(): new_val for new_val, old_val in enumerate(unique_old_dim_indices)
    }

    indices[dim] = torch.tensor([
        new_dim_map[idx.item()] for idx in old_dim_indices
    ])
    size[dim] = unique_old_dim_indices.size()[0]

    return torch.sparse_coo_tensor(indices, tensor.coalesce().values(), size, 
                                   is_coalesced=tensor.is_coalesced())


def slice(tensor : torch.Tensor, 
          key : int | slice | torch.Tensor,
          dim : int = 0,
          invert : bool = False) -> torch.Tensor:
    """
    Slice of a sparse tensor in COO format by key in a given dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Sparse tensor in COO format
    key : int | slice | torch.Tensor
        Key for index selection
    dim : int
        Dimension
        Default: 0
    invert : bool
        Select all indexes that not match the key
    
    Returns
    -------
        torch.Tensor
    """
    return clear(reset(tensor, key, dim, invert), dim)


def cat(tensor1 : torch.Tensor, 
        tensor2 : torch.Tensor, 
        dim : int = 0) -> torch.Tensor:
    """
    Concatenate two sparse torch tensor in coo format along dimension

    Parameters
    ----------
    tensor1 : torch.Tensor
        Sparse tensor in COO format
    tensor2 : torch.Tensor
        Sparse tensor in COO format
    dim : int
        Dimension
        Default: 0
    
    Returns
    -------
        torch.Tensor
    """
    # Adjust indices for the second tensor
    shifted_indices = tensor2.indices().clone()
    shifted_indices[dim, :] += tensor1.size()[dim]

    # Concatenate indices and values
    concatenated_indices = torch.cat((tensor1.indices(), shifted_indices), dim=1)
    concatenated_values = torch.cat((tensor1.values(), tensor2.values()))

    # Determine new size
    new_size = list(tensor1.size())
    new_size[dim] += tensor2.size()[dim]

    # Create the new sparse tensor
    return torch.sparse_coo_tensor(concatenated_indices, concatenated_values, new_size)


def equal(t1 : torch.Tensor, t2 : torch.Tensor) -> bool:
    if t1.size() != t2.size():
        return False
    if t1._nnz() != t2._nnz():
        return False
    if not torch.equal(t1._indices(), t2._indices()):
        return False
    if not torch.equal(t1._values(), t2._values()):
        return False
    return True
