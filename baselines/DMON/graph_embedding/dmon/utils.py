# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for graph processing."""
import numpy as np
import scipy.sparse
from scipy.sparse import base
import tensorflow as tf


def normalize_graph(graph,
                    normalized = True,
                    add_self_loops = True):
  """Normalized the graph's adjacency matrix in the scipy sparse matrix format.

  Args:
    graph: A scipy sparse adjacency matrix of the input graph.
    normalized: If True, uses the normalized Laplacian formulation. Otherwise,
      use the unnormalized Laplacian construction.
    add_self_loops: If True, adds a one-diagonal corresponding to self-loops in
      the graph.

  Returns:
    A scipy sparse matrix containing the normalized version of the input graph.
  """
  if add_self_loops:
    graph = graph + scipy.sparse.identity(graph.shape[0])
  degree = np.squeeze(np.asarray(graph.sum(axis=1)))
  if normalized:
    with np.errstate(divide='ignore'):
      inverse_sqrt_degree = 1. / np.sqrt(degree)
    inverse_sqrt_degree[inverse_sqrt_degree == np.inf] = 0
    inverse_sqrt_degree = scipy.sparse.diags(inverse_sqrt_degree)
    return inverse_sqrt_degree @ graph @ inverse_sqrt_degree
  else:
    with np.errstate(divide='ignore'):
      inverse_degree = 1. / degree
    inverse_degree[inverse_degree == np.inf] = 0
    inverse_degree = scipy.sparse.diags(inverse_degree)
    return inverse_degree @ graph
  
def normalize_graph_tf_sparse(graph_tf_sparse,
                                       normalized=True,
                                       add_self_loops=True):
    """Normalizes the graph's adjacency matrix in tf.SparseTensor format.

    Args:
        graph_tf_sparse: A tf.SparseTensor adjacency matrix of the input graph.
        normalized: If True, uses the normalized Laplacian formulation.
        add_self_loops: If True, adds self-loops to the graph.

    Returns:
        A tf.SparseTensor containing the normalized version of the input graph.
    """
    n_nodes = graph_tf_sparse.dense_shape[0]
    
    # Add self-loops if requested
    if add_self_loops:
        # Create self-loops indices and values
        self_loop_indices = tf.range(n_nodes, dtype=tf.int64)
        self_loop_indices = tf.stack([self_loop_indices, self_loop_indices], axis=1)
        self_loop_values = tf.ones([n_nodes], dtype=graph_tf_sparse.values.dtype)
        
        # Combine with original graph
        all_indices = tf.concat([graph_tf_sparse.indices, self_loop_indices], axis=0)
        all_values = tf.concat([graph_tf_sparse.values, self_loop_values], axis=0)
        
        graph_with_loops = tf.sparse.SparseTensor(
            indices=all_indices,
            values=all_values,
            dense_shape=graph_tf_sparse.dense_shape
        )
        graph_with_loops = tf.sparse.reorder(graph_with_loops)
    else:
        graph_with_loops = graph_tf_sparse
    
    # Calculate degree from sparse tensor
    degree = tf.sparse.reduce_sum(graph_with_loops, axis=1)
    
    if normalized:
        # Normalized Laplacian
        inverse_sqrt_degree = tf.where(degree > 0, 
                                      1.0 / tf.sqrt(degree), 
                                      0.0)
        
        # Normalize by multiplying rows and columns with inverse_sqrt_degree
        row_scale = tf.gather(inverse_sqrt_degree, graph_with_loops.indices[:, 0])
        col_scale = tf.gather(inverse_sqrt_degree, graph_with_loops.indices[:, 1])
        
        normalized_values = graph_with_loops.values * row_scale * col_scale
        
        normalized_graph = tf.sparse.SparseTensor(
            indices=graph_with_loops.indices,
            values=normalized_values,
            dense_shape=graph_with_loops.dense_shape
        )
        
    else:
        # Unnormalized Laplacian
        inverse_degree = tf.where(degree > 0, 
                                 1.0 / degree, 
                                 0.0)
        
        # Normalize by multiplying rows with inverse_degree
        row_scale = tf.gather(inverse_degree, graph_with_loops.indices[:, 0])
        
        normalized_values = graph_with_loops.values * row_scale
        
        normalized_graph = tf.sparse.SparseTensor(
            indices=graph_with_loops.indices,
            values=normalized_values,
            dense_shape=graph_with_loops.dense_shape
        )
    
    return tf.sparse.reorder(normalized_graph)

