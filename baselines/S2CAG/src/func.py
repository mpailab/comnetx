# -*- coding: utf-8 -*-
import time
import warnings
import numpy as np
from sklearn.utils import check_random_state, as_float_array
from scipy.sparse import csc_matrix
from sklearn import preprocessing
from scipy import linalg, sparse
# from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip, safe_sparse_dot




def SNEM_rounding(vectors, T=100):
    vectors = as_float_array(vectors)
    n_samples = vectors.shape[0]
    n_feats = vectors.shape[1]

    labels = vectors.argmax(axis=1)
    # print(type(labels), labels.shape)
    vectors_discrete = csc_matrix(
            (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
            shape=(n_samples, n_feats))

    vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
    vectors_sum[vectors_sum==0]=1
    vectors_discrete = vectors_discrete*1.0/vectors_sum
    #vectors_discrete = preprocessing.normalize(vectors_discrete, norm='l2', axis=0)

    for _ in range(T):
        Q = vectors.T.dot(vectors_discrete)
        Q=np.matrix(Q)
        Q=np.asarray(Q)
        vectors_discrete = vectors.dot(Q)
        vectors_discrete = as_float_array(vectors_discrete)

        labels = vectors_discrete.argmax(axis=1)
        vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_feats))

        
        #vectors_discrete = preprocessing.normalize(vectors_discrete, norm='l2', axis=0)

        vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
        vectors_sum[vectors_sum==0]=1
        vectors_discrete = vectors_discrete*1.0/vectors_sum

    return labels


def KSI_decompose_B(Z, dim, tau=100,gamma=0.9):
    random_state = 0
    random_state = check_random_state(random_state)
    Z = adapt(Z)
    # print(Z)
    n = Z.shape[0]
    Q = random_state.normal(size=(n, dim))
    one_vec = np.ones((n, 1))
    Q, _ = linalg.qr(Q, mode='economic')
    D_Z_2 = Z.T.dot(one_vec)
    D_Z_3 = Z.dot(D_Z_2)
    D_sum = D_Z_3.sum()

    for i in range(tau):
        Z_T=Z.T.dot(Q)
        A_Z=Z.dot(Z_T)
        D_Z_1 = one_vec.T.dot(A_Z)

        DD_Z=D_Z_3.dot(D_Z_1)
        B = A_Z - (DD_Z / D_sum) * gamma
        Q,_=linalg.qr(B, mode='economic')

    return Q
def adapt(Z):
    D=Z.sum(1).reshape((Z.shape[0],1))
    D[D<0]=0
    l_vec=np.ones((Z.shape[0],1))
    # D=D+l_vec
    D=np.sqrt(D)
    # D=np.log(D+l_vec)


    Z=Z*D
    return Z

def subspace_svd(
    A, X,
    n_components,
    *,
    n_iter="auto",
    n_T=1,
    alpha=0.5,
    flip_sign=True,
    random_state="warn",
    n_oversamples=10,
):
    if isinstance(A, (sparse.lil_matrix, sparse.dok_matrix)):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(F).__name__),
            sparse.SparseEfficiencyWarning,
        )

    if isinstance(X, (sparse.lil_matrix, sparse.dok_matrix)):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(B).__name__),
            sparse.SparseEfficiencyWarning,
        )

    if random_state == "warn":
        warnings.warn(
            "If 'random_state' is not supplied, the current default "
            "is to use 0 as a fixed seed. This will change to  "
            "None in version 1.2 leading to non-deterministic results "
            "that better reflect nature of the randomized_svd solver. "
            "If you want to silence this warning, set 'random_state' "
            "to an integer seed or to None explicitly depending "
            "if you want your code to be deterministic or not.",
            FutureWarning,
        )
        random_state = 0

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples

    if n_iter == "auto":
        n_iter = 7 if n_components < 0.1 * min(X.shape) else 4



    # Generating normal random vectors with shape: (A.shape[1], size)
    G = random_state.normal(size=(X.shape[1], n_random))
    if X.dtype.kind == "f":
        # Ensure f32 is preserved as f32
        G = G.astype(X.dtype, copy=False)

    Q = G

    # I = sparse.identity(A.shape[0])
    factor = n_T if alpha==1.0 else (1.-alpha)/(1.-np.power(alpha, n_T+1))

    for i in range(n_iter):
        Q = safe_sparse_dot(X, Q)
        Q0 = Q.copy()
        for j in range(n_T):
            Q = alpha*safe_sparse_dot(A, Q) + Q0
            # Q = safe_sparse_dot(A, Q)

        Q = factor*Q

        Q0 = Q.copy()
        for j in range(n_T):
            Q = alpha*safe_sparse_dot(A.T, Q) + Q0
            # Q = safe_sparse_dot(A.T, Q)

        Q = factor*Q

        Q = safe_sparse_dot(X.T, Q)


    Q = safe_sparse_dot(X, Q)
    Q0 = Q.copy()
    for j in range(n_T):
        Q = alpha*safe_sparse_dot(A, Q) + Q0
        # Q = safe_sparse_dot(A, Q)

    Q = factor*Q

    Q, _ = linalg.qr(Q, mode="economic")



    # project M to the (k + p) dimensional space using the basis vectors
    B = Q.copy()
    for j in range(n_T):
        B = alpha*safe_sparse_dot(A.T, B) + Q
    B=factor*B
    B = safe_sparse_dot(X.T, B).T

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, Vt = linalg.svd(B, full_matrices=False, lapack_driver="gesdd")


    del B

    U = np.dot(Q, Uhat)

    if flip_sign:
        U, Vt = svd_flip(U, Vt)

    return U[:, 1:n_components+1] #, s[:n_components], Vt[:n_components, :]

def sub_randomized_svd(
    Z,
    n_components,
    *,
    n_iter="auto",
    flip_sign=True,
    random_state="warn",
    n_oversamples=10,
):
    if isinstance(Z, (sparse.lil_matrix, sparse.dok_matrix)):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(F).__name__),
            sparse.SparseEfficiencyWarning,
        )



    if random_state == "warn":
        warnings.warn(
            "If 'random_state' is not supplied, the current default "
            "is to use 0 as a fixed seed. This will change to  "
            "None in version 1.2 leading to non-deterministic results "
            "that better reflect nature of the randomized_svd solver. "
            "If you want to silence this warning, set 'random_state' "
            "to an integer seed or to None explicitly depending "
            "if you want your code to be deterministic or not.",
            FutureWarning,
        )
        random_state = 0

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples

    if n_iter == "auto":
        n_iter = 7 if n_components < 0.1 * min(Z.shape) else 4



    # Generating normal random vectors with shape: (A.shape[1], size)
    G = random_state.normal(size=(Z.shape[1], n_random))
    if Z.dtype.kind == "f":
        # Ensure f32 is preserved as f32
        G = G.astype(Z.dtype, copy=False)

    Q = G

    # I = sparse.identity(A.shape[0])

    for i in range(n_iter):
        Q=safe_sparse_dot(Z,Q)
        Q=safe_sparse_dot(Z.T,Q)
    Q = safe_sparse_dot(Z, Q)

    # print('Q =', Q)
    Q, _ = linalg.qr(Q, mode="economic")
    

    # project M to the (k + p) dimensional space using the basis vectors
    B=safe_sparse_dot(Z.T,Q).T
    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, Vt = linalg.svd(B, full_matrices=False, lapack_driver="gesdd")

    del B

    U = np.dot(Q, Uhat)

    if flip_sign:
        U, Vt = svd_flip(U, Vt)

    return U[:, 1:n_components+1] #, s[:n_components], Vt[:n_components, :]
