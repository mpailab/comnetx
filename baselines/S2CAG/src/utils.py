from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfTransformer
from time import time
from scipy import sparse
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')

# from ogb.nodeproppred import NodePropPredDataset
import numpy as np
from scipy.sparse import csr_matrix
import scipy.io as io
import os

from sklearn.utils.extmath import randomized_svd
from sklearn import preprocessing

from func import  SNEM_rounding, subspace_svd,KSI_decompose_B,sub_randomized_svd


def datagen(dataset):
  if dataset in ['wiki', 'pubmed', 'computers', 'acm', 'dblp', 'citeseer', 'corafull','arxiv']:
    data = io.loadmat(os.path.join('data', f'{dataset}.mat'))
    features = data['fea'].astype(float)
    adj = data.get('W')


    if adj is not None:
      adj = adj.astype(float)
      if not sp.issparse(adj):
          adj = sp.csc_matrix(adj)


    if not sparse and sp.issparse(features):
        features = features.toarray()
    if dataset in ['wiki', 'pubmed', 'computers', 'citeseer', 'corafull','arxiv']:
        labels = data['gnd'].reshape(-1) - 1
    else:
        labels=data['gnd'].reshape(-1)
    n_classes = len(np.unique(labels))

    return adj, features, labels, n_classes

    return adj, features, labels, n_classes
  if dataset in ['Amazon_photos']:
      data = io.loadmat(os.path.join('data', f'{dataset}.mat'))
      features = data['features'].astype(float)
      adj = data.get('adj')
      if adj is not None:
          adj = adj.astype(float)
          if not sp.issparse(adj):
              adj = sp.csc_matrix(adj)

      if not sparse and sp.issparse(features):
          features = features.toarray()
      labels = data['label'].reshape(-1)
      n_classes = len(np.unique(labels))
      return adj, features, labels, n_classes
     

def preprocess_dataset(adj, features, row_norm=True, sym_norm=True, feat_norm='l2', tf_idf=False, sparse=False, alpha=1, beta=1):

    if sym_norm:
        adj = aug_normalized_adjacency(adj, True, alpha=alpha)
    if row_norm:
        adj = row_normalize(adj, True, alpha=beta)
        # adj=similarity_M(adj,True)


    if tf_idf:
        features = TfidfTransformer(norm=feat_norm).fit_transform(features)
    else:
        features = normalize(features, feat_norm)
    
    if not sparse:
        features = features.toarray()
    return adj, features

def aug_normalized_adjacency(adj, add_loops=True, alpha=1):
    if add_loops:
        adj = adj + alpha*sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def row_normalize(mx, add_loops=True, alpha=1):
    if add_loops:
        mx = mx + alpha * sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def convert_sparse_matrix_to_sparse_tensor(X):
      coo = X.tocoo()
      indices = np.mat([coo.row, coo.col]).transpose()
      return tf.SparseTensor(indices, coo.data, coo.shape)


def clustering_accuracy(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    def ordered_confusion_matrix(y_true, y_pred):
      conf_mat = confusion_matrix(y_true, y_pred)
      w = np.max(conf_mat) - conf_mat
      row_ind, col_ind = linear_sum_assignment(w)
      conf_mat = conf_mat[row_ind, :]
      conf_mat = conf_mat[:, col_ind]
      return conf_mat

    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


# @tf.function
def PowerIteration(feature, adj_normalized, power, alpha):
  #feature = preprocessing.normalize(feature, norm='l2', axis=1)
  print('PowerIteration!!!!!!!!!!!')
  feat0 = feature.copy()
  factor = power if alpha == 1.0 else (1.-alpha)/ (1. - np.power(alpha, power + 1))
  print('after factor !!!!!!!!!!!')
  for _ in range(power):
    print('PowerIteration in range!!!!!')
    feature = alpha*(adj_normalized.dot(feature)+feat0)
    print('power!!!!!', power)

  feature=factor*feature
  print('after factor*feature!!!!!!!!!')
  return feature



def run_SSCAG(X, k, adj_normalized, T, alpha,method="sub",dataset='acm',gamma=0.9,tau=7):
  print('run_SSCAG!!!!!!!!')
  x = X.sum(0)
  D = X @ x
  D[D==0]=1

  D = np.sqrt(D)
  D = sparse.diags(1.0/D)
  X = D.dot(X)
  n = X.shape[0]
  d = X.shape[1]
  m=adj_normalized.nnz

  if method =='sub':
     integr=m*d*T+2*(tau+1)*n*d*(k+10)+3*n*(k+10)**2
     naive=2*(tau+1)*(n*d+m*T)*(k+10)+3*n*(k+10)**2
     if integr<naive:
         Z = PowerIteration(X, adj_normalized, T, alpha)
         print('After Z!!!!!!!!!!!')
         Q= sub_randomized_svd(Z, n_components=k,n_iter=tau)

     else:
        Q = subspace_svd(adj_normalized, X, n_components=k, n_iter=tau,n_T=T, alpha=alpha)

  elif method == 'mod':
          Z = PowerIteration(X, adj_normalized, T, alpha)
          Q = KSI_decompose_B(Z=Z, dim=k,tau=tau,gamma=gamma)
  P = SNEM_rounding(Q)
  return P, Q
