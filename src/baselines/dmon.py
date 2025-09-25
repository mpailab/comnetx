from typing import Tuple
import sys
from absl import app
from absl import flags
import os
import numpy as np
import scipy.sparse
from scipy.sparse import base
import sklearn.metrics
import torch
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter("ignore")

stdout = sys.stdout
stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()

sys.stdout.close()
sys.stderr.close()
sys.stdout = stdout
sys.stderr = stderr

tf.get_logger().setLevel('ERROR')

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
dmon_root = os.path.join(PROJECT_PATH, "baselines", "DMON")
if dmon_root not in sys.path:
   sys.path.insert(0, dmon_root)

from graph_embedding.dmon import dmon
from graph_embedding.dmon import gcn
from graph_embedding.dmon import metrics
from graph_embedding.dmon import utils


def copy_sparse_tensor(sparse_tensor):
    return tf.SparseTensor(
        indices=sparse_tensor.indices,
        values=sparse_tensor.values,
        dense_shape=sparse_tensor.dense_shape
    )

def torch_to_scipy_csr(tensor):
    if tensor.is_sparse:
        indices = tensor.indices().cpu()
        values = tensor.values().detach().cpu()
        shape = tensor.shape
        
        coo_matrix = scipy.sparse.coo_matrix((values.numpy(), (indices[0].numpy(), indices[1].numpy())), shape=shape)
        return coo_matrix.tocsr()
    
    else:
        numpy_array = tensor.detach().cpu().numpy()
        return scipy.sparse.csr_matrix(numpy_array)

def convert_scipy_sparse_to_sparse_tensor(matrix):
  """Converts a sparse matrix and converts it to Tensorflow SparseTensor.

  Args:
    matrix: A scipy sparse matrix.

  Returns:
    A ternsorflow sparse matrix (rank-2 tensor).
  """
  matrix = matrix.tocoo()
  return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)
   
def torch_to_tf_sparse_tensor(tensor):
    """
    Converts a TorchTensor to Tensorflow SparseTensor

    Args:
      tensor: A Torch tensor.

    Returns:
      A ternsorflow sparse matrix (rank-2 tensor).
    """
    if tensor.is_sparse:
        indices = tensor.indices().cpu().numpy().T  # [i, j] format
        values = tensor.values().detach().cpu().numpy().astype(np.float32)
        shape = tensor.shape
        
        return tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=shape
        )
    
    # For dense tensor
    else:
        numpy_array = tensor.detach().cpu().numpy()
        
        # Create sparse tensor from dense array
        nonzero_indices = np.argwhere(numpy_array != 0)
        nonzero_values = numpy_array[nonzero_indices[:, 0], nonzero_indices[:, 1]].astype(np.float32)
        
        return tf.sparse.SparseTensor(
            indices=nonzero_indices,
            values=nonzero_values,
            dense_shape=numpy_array.shape
        )

def build_dmon(input_features,
               input_graph,
               input_adjacency,
               args):
  """Builds a Deep Modularity Network (DMoN) model from the Keras inputs.

  Args:
    input_features: A dense [n, d] Keras input for the node features.
    input_graph: A sparse [n, n] Keras input for the normalized graph.
    input_adjacency: A sparse [n, n] Keras input for the graph adjacency.

  Returns:
    Built Keras DMoN model.
  """
  output = input_features
  for n_channels in args._architecture:
    output = gcn.GCN(n_channels)([output, input_graph])
  pool, pool_assignment = dmon.DMoN(
      args._n_clusters,
      collapse_regularization=args._collapse_regularization,
      dropout_rate=args._dropout_rate)([output, input_adjacency])
  return tf.keras.Model(
      inputs=[input_features, input_graph, input_adjacency],
      outputs=[pool, pool_assignment])


def adapted_dmon(adj: torch.Tensor,
        ftrs : torch.Tensor,
        lbls : torch.Tensor | None = None,
        args=None,
        **kwargs):
  """
  DMON method

  Parameters
    ----------
    adj : torch.Tensor
        Adjacency matrix, shape [N, N].

    ftrs: torch.Tensor
        Features matrix, shape [N, K].

    lbls: torch.Tensor or None, optional
        Ground-truth node labels.
        Default: None 
    args: 
        Hyperparameters for DMON training. If None, default parameters are used.
  """
  if args is None:
    class Args:
        _architecture = [64]
        _collapse_regularization = 1
        _dropout_rate = 0 #min - 0, max - 1
        _n_clusters = 16 #min - 0
        _n_epochs = 1000 #min - 0
        _learning_rate = 0.001 #min - 0
    args = Args()
  for key, value in kwargs.items():
    if hasattr(args, key):
        setattr(args, key, value)
    else:
        raise ValueError(f"Unknown argument {key}")

  graph, features = torch_to_tf_sparse_tensor(adj), torch_to_tf_sparse_tensor(ftrs)
  adjacency = torch_to_scipy_csr(adj)
  if lbls != None:
    labels_numpy = lbls.detach().cpu().numpy().flatten()
    label_indices = np.where(labels_numpy != -1)[0]
    know_labels = labels_numpy[label_indices]

  # features = features.todense()
  # n_nodes = adjacency.shape[0]
  # feature_size = features.shape[1]
  features_dense = tf.sparse.to_dense(features) if isinstance(features, tf.SparseTensor) else features
  n_nodes = tf.shape(graph)[0]
  feature_size = tf.shape(features_dense)[1]
  graph_normalized = utils.normalize_graph_tf_sparse(copy_sparse_tensor(graph))
  
  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = build_dmon(input_features, input_graph, input_adjacency, args)

  def grad(model, inputs):
    with tf.GradientTape() as tape:
      _ = model(inputs, training=True)
      loss_value = sum(model.losses)
    return model.losses, tape.gradient(loss_value, model.trainable_variables)
  
  optimizer = tf.keras.optimizers.Adam(args._learning_rate)
  model.compile(optimizer, None)

  for epoch in range(args._n_epochs):
    loss_values, grads = grad(model, [features, graph_normalized, graph])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'epoch {epoch}, losses: ' +
         ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))

  # Obtain the cluster assignments.
  _, assignments = model([features, graph_normalized, graph], training=False)
  assignments = assignments.numpy()
  print("Assignments shape:", assignments.shape)  
  print("Assignments type:", assignments.dtype)  

  clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
  print("Clusters shape:", clusters.shape)        
  print("Clusters type:", clusters.dtype)         
  print("Unique clusters:", np.unique(clusters))  
  print("Cluster sizes:", np.bincount(clusters))  
  if not isinstance(clusters, torch.Tensor):
    clusters_tensor = torch.from_numpy(clusters).long()
  else:
    clusters_tensor = clusters.long()
  # Prints some metrics used in the paper.
  print('Conductance:', metrics.conductance(adjacency, clusters))
  print('Modularity:', metrics.modularity(adjacency, clusters))
  if lbls != None:
     print(
      'NMI:',
      sklearn.metrics.normalized_mutual_info_score(
          know_labels, clusters[label_indices], average_method='arithmetic'))
     precision = metrics.pairwise_precision(know_labels, clusters[label_indices])
     recall = metrics.pairwise_recall(know_labels, clusters[label_indices])
     print('F1:', 2 * precision * recall / (precision + recall))
  print("Returning clusters_tensor:", type(clusters_tensor), clusters_tensor.shape)
  return clusters_tensor
  
