import torch
import random
import numpy as np
import scipy.sparse as sparse
import argparse
import time
np.random.seed((3,14159))


parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=True, help='print debug messages')
parser.add_argument('--n', type=int, default=1000000, help='number of vertices in matrix')
parser.add_argument('--clust', type=int, default=3, help='number of clusters')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dencity', type=float, default=0.001)
args = parser.parse_args()

def sparse_conv(Q, ind1, ind2, device):
    #print(Q.to_dense())
    crows = [i for i in range(ind1+1)] + [ind2] + [i for i in range(ind2+1, Q.size()[0]+1)]
    cols = [i for i in range(Q.size()[0])]
    val = [1.]*Q.size()[0]
    conv_mtrx = torch.sparse_csr_tensor(crow_indices = crows, col_indices = cols, values = val,
                                        size = (Q.size()[0] - ind2 + ind1+1, Q.size()[0]), dtype=torch.double,
                                        device=device)
    before_mm = time.time()
    #print(conv_mtrx.to_dense())
    left_mm = torch.sparse.mm(conv_mtrx, Q)
    conv_mtrx_t = torch.transpose(conv_mtrx, 0, 1).to_sparse_csr()
    result = torch.sparse.mm(left_mm, conv_mtrx_t)
    mm_time = time.time() - before_mm
    print(f"Matmul time: {mm_time:.6f} seconds")
    return result

def alt_gen(n, numb_clust, device):
    csr_mat = sparse.random(n, n, density=0.001, format='csr',  rng=None, dtype=None, data_rvs=None)
    upper_csr_mat = sparse.triu(csr_mat)
    csr_mat = upper_csr_mat + upper_csr_mat.T - sparse.diags(csr_mat.diagonal())
    values = csr_mat.data
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    shape = csr_mat.shape
    result = torch.sparse_csr_tensor(crow_indices = indptr, col_indices = indices, values = values, size = shape,
                                     dtype=torch.double, device=device)
    nmbr_vrtcs_clstr = n // numb_clust
    cluster_sizes = [random.randint(1, nmbr_vrtcs_clstr) for _ in range(numb_clust)]
    return result, cluster_sizes


def generate_random_csr_tensor(n, nnz, numb_clust, dtype=torch.double, device='cpu'):
    assert nnz <= n * n, "Too much non-zero elements"

    coords = set()
    while len(coords) < nnz:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        coords.add((i, j))

    coords = sorted(coords)  # важно для CSR
    row_indices = [c[0] for c in coords]
    col_indices = [c[1] for c in coords]
    values = torch.randn(len(coords), dtype=dtype, device=device)

    crow_indices = torch.zeros(n + 1, dtype=torch.int32, device=device)
    for r in row_indices:
        crow_indices[r + 1] += 1
    crow_indices = torch.cumsum(crow_indices, dim=0)

    col_indices = torch.tensor(col_indices, dtype=torch.int32, device=device)
    csr_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(n, n), device=device)

    nmbr_vrtcs_clstr = n // numb_clust
    cluster_sizes = [random.randint(1, nmbr_vrtcs_clstr) for _ in range(numb_clust)]
    return csr_tensor, cluster_sizes

if __name__ == '__main__':
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("cuda is not available")
    else:
        device = args.device
    print('Using device:', device)
    start_time = time.time()
    #Q, cluster_sizes= alt_gen(args.n, args.clust, device)
    Q, cluster_sizes = generate_random_csr_tensor(n = args.n, nnz = int(args.n*args.dencity), numb_clust=args.clust, device = args.device)
    after_gen_time = time.time()
    gen_time = after_gen_time - start_time
    print(f"Generate time: {gen_time:.6f} seconds")
    print("Matrix size before conv:", Q.size())
    print("Indexes of cluster ", cluster_sizes[0], cluster_sizes[0] + cluster_sizes[1]-1)
    Q = sparse_conv(Q, cluster_sizes[0], cluster_sizes[0] + cluster_sizes[1], device)
    #print(Q.to_dense())
    conv_time = time.time() - after_gen_time
    print(f"Conv time: {conv_time:.6f} seconds")
    print("Matrix size after conv:", Q.size())
    print(args.device)


