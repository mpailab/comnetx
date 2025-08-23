import torch


def reset_columns_coo_tensor(coo_tensor : torch.Tensor, 
                             indices : torch.Tensor, 
                             invert : bool = False):
    """
    Reset columns of sparse torch tensor in coo format by indices

    Parameters
    ----------
    coo_tensor : torch.sparse_coo
        Tensor of the shape (k x n)
    indices : torch.Tensor
        List of column indices (n, )
    invert : bool
        Reset all columns whose indexes do not belong to indices
    
    Returns
    -------
        torch.sparse_coo
    """
    column_indices_of_elements = coo_tensor.indices()[1]
    mask = torch.isin(column_indices_of_elements, indices)
    if invert:
        mask = ~mask
    return torch.sparse_coo_tensor(coo_tensor.indices()[:, mask], 
                                   coo_tensor.values()[mask],
                                   coo_tensor.size()).coalesce()


def reset_rows_coo_tensor(coo_tensor : torch.Tensor, 
                          indices : torch.Tensor, 
                          invert : bool = False):
    """
    Reset rows of sparse torch tensor in coo format by indices

    Parameters
    ----------
    coo_tensor : torch.sparse_coo
        Tensor of the shape (k x n)
    indices : torch.Tensor
        List of row indices (n, )
    invert : bool
        Reset all rows whose indexes do not belong to indices
    
    Returns
    -------
        torch.sparse_coo
    """
    row_indices_of_elements = coo_tensor.indices()[0]
    mask = torch.isin(row_indices_of_elements, indices)
    if invert:
        mask = ~mask
    return torch.sparse_coo_tensor(coo_tensor.indices()[mask, :], 
                                   coo_tensor.values()[mask],
                                   coo_tensor.size()).coalesce()


def remove_zero_columns(coo_tensor : torch.Tensor):
    """
    Remove zero columns from sparse torch tensor in coo format

    Parameters
    ----------
    coo_tensor : torch.sparse_coo
        Sparse coo tensor
    
    Returns
    -------
        torch.sparse_coo
    """
    # Reindexing
    old_indices = coo_tensor.indices()
    old_column_indices = old_indices[1, :]
    unique_old_columns = torch.unique(old_column_indices)
    new_column_map = {
        old_val.item(): new_val for new_val, old_val in enumerate(unique_old_columns)
    }

    # Apply mapping to column indices
    reindexed_column_indices = torch.tensor([
        new_column_map[idx.item()] for idx in old_column_indices
    ])

    # Combine reindexed column indices with original column indices
    new_indices = torch.stack([old_indices[0, :], reindexed_column_indices])
    new_size = [coo_tensor.size()[0], unique_old_columns.size()[0]]

    # Create the new sparse tensor
    reindexed_coo_tensor = torch.sparse_coo_tensor(new_indices,
                                                   coo_tensor.values(),
                                                   new_indices.size()).coalesce()

    return reindexed_coo_tensor


def remove_zero_rows(coo_tensor : torch.Tensor):
    """
    Remove zero rows from sparse torch tensor in coo format

    Parameters
    ----------
    coo_tensor : torch.sparse_coo
        Sparse coo tensor
    
    Returns
    -------
        torch.sparse_coo
    """
    # Reindexing
    old_indices = coo_tensor.indices()
    old_row_indices = old_indices[0, :]
    unique_old_rows = torch.unique(old_row_indices)
    new_row_map = {
        old_val.item(): new_val for new_val, old_val in enumerate(unique_old_rows)
    }

    # Apply mapping to row indices
    reindexed_row_indices = torch.tensor([
        new_row_map[idx.item()] for idx in old_row_indices
    ])

    # Combine reindexed row indices with original column indices
    new_indices = torch.stack([reindexed_row_indices, old_indices[1, :]])
    new_size = [unique_old_rows.size()[0], coo_tensor.size()[1]]

    # Create the new sparse tensor
    reindexed_coo_tensor = torch.sparse_coo_tensor(new_indices,
                                                   coo_tensor.values(),
                                                   new_size).coalesce()

    return reindexed_coo_tensor


def slice_columns_coo_tensor(coo_tensor : torch.Tensor, indices : torch.Tensor):
    """
    Slice columns of sparse torch tensor in coo format by indices

    Parameters
    ----------
    coo_tensor : torch.sparse_coo
        Tensor of the shape (k x n)
    indices : torch.Tensor
        List of column indices (n, )
    
    Returns
    -------
        torch.sparse_coo
    """
    return remove_zero_columns(reset_columns_coo_tensor(coo_tensor, indices, invert=True))


def slice_rows_coo_tensor(coo_tensor : torch.Tensor, indices : torch.Tensor):
    """
    Slice rows of sparse torch tensor in coo format by indices

    Parameters
    ----------
    coo_tensor : torch.sparse_coo
        Tensor of the shape (k x n)
    indices : torch.Tensor
        List of row indices (n, )
    
    Returns
    -------
        torch.sparse_coo
    """
    return remove_zero_rows(reset_rows_coo_tensor(coo_tensor, indices, invert=True))


def cat_coo_tensors(coo_tensor1 : torch.Tensor, coo_tensor2 : torch.Tensor, dim : int = 0):
    """
    Concatenate two sparse torch tensor in coo format along dimension

    Parameters
    ----------
    coo_tensor1 : torch.sparse_coo
        Sparse coo tensor
    coo_tensor2 : torch.sparse_coo
        Sparse coo tensor
    dim : int
        Dimension
        Default: 0
    
    Returns
    -------
        torch.sparse_coo
    """
    # Adjust indices for the second tensor
    shifted_indices = coo_tensor2.indices().clone()
    shifted_indices[dim, :] += coo_tensor1.size()[dim]

    # Concatenate indices and values
    concatenated_indices = torch.cat((coo_tensor1.indices(), shifted_indices), dim=1)
    concatenated_values = torch.cat((coo_tensor1.values(), coo_tensor2.values()))

    # Determine new size
    new_size = list(coo_tensor1.size())
    new_size[dim] += coo_tensor2.size()[dim]

    # Create the new sparse tensor
    return torch.sparse_coo_tensor(concatenated_indices, concatenated_values, new_size)


def are_coo_tensors_equal(t1, t2):
    if t1.size() != t2.size():
        return False
    if t1._nnz() != t2._nnz():
        return False
    if not torch.equal(t1._indices(), t2._indices()):
        return False
    if not torch.equal(t1._values(), t2._values()):
        return False
    return True


def add_zero_lines_and_columns(matrix : torch.Tensor, line_num : int, col_num : int):
    """Add to sparse matrix new lines and columns (filled by zeros)"""
    zero_columns = torch.sparse_coo_tensor(size=(matrix.size()[0], col_num))
    matrix = torch.cat((matrix, zero_columns), dim=1)
    zero_lines = torch.sparse_coo_tensor(size=(line_num, matrix.size()[1]))
    matrix = torch.cat((matrix, zero_lines), dim=0)
    return matrix

def to_submatrix(matrix : torch.Tensor, lmask, cmask):
    """
    Go to submatrix

    Parameters
    ----------
    matrix (torch.sparse_coo) : matrix (k x n)
    lmask (torch.Tensor) : mask for lines (k, )
    cmask (torch.Tensor) : mask for columns (n, )
    
    Return:
        torch.sparse_coo: submatrix
    """
    # FIXME don't use to_dense()
    return matrix.to_dense()[:, cmask][lmask].to_sparse()

def sparse_eye(n : int, dtype=int):
    """Create eye sparse matrix"""
    w = torch.ones(n, dtype=dtype)
    i = torch.stack((torch.arange(n), torch.arange(n)))
    return torch.sparse_coo_tensor(i, w, size=(n, n))

def bool_mm(t1 : torch.Tensor, t2 : torch.Tensor):
    """Matrix multiplication for sparse bool matrices"""
    t1_int = t1.type(torch.int8)
    t2_int = t2.type(torch.int8)
    return torch.sparse.mm(t1_int, t2_int).bool()