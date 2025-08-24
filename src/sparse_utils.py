import torch


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