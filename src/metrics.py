import torch
import tensorflow as tf
import time

class Metrics:

    def __init__(self):
        pass

    def modularity(adjacency, assignments):
        assignments_pool = assignments / tf.math.reduce_sum(assignments, axis=0)
        
        degrees = tf.sparse.reduce_sum(adjacency, axis=0)
        degrees = tf.reshape(degrees, (-1, 1))
        m = tf.math.reduce_sum(degrees)  
        
        graph_pooled = tf.transpose(
            tf.sparse.sparse_dense_matmul(adjacency, assignments))
        graph_pooled = tf.matmul(graph_pooled, assignments)
        
        ca = tf.matmul(assignments, degrees, transpose_a=True)
        cb = tf.matmul(degrees, assignments, transpose_a=True)
        normalizer = tf.matmul(ca, cb) / 2 / m
        
        modularity = tf.linalg.trace(graph_pooled - normalizer) / 2 / m
        
        return modularity

    def fast_modularity(adjacency, assignments):
        degrees = tf.sparse.reduce_sum(adjacency, axis=0)
        m = tf.reduce_sum(degrees)
        inv_2m = 1.0 / (2 * m) 
        degrees = tf.reshape(degrees, (-1, 1))
        
        as_product = tf.sparse.sparse_dense_matmul(adjacency, assignments)
        
        graph_pooled = tf.matmul(as_product, assignments, transpose_a=True)
        
        s_times_d = tf.matmul(assignments, degrees, transpose_a=True)
        normalizer = tf.matmul(s_times_d, s_times_d, transpose_b=True) * inv_2m
        
        modularity = tf.linalg.trace(graph_pooled - normalizer) * inv_2m
        
        return modularity

if __name__ == "__main__":
    N, K = 1000000, 100
    
    # Data generation
    features = tf.random.normal((N, 10))
    indices = tf.random.uniform((200, 2), 0, N, dtype=tf.int64)
    values = tf.ones(200)
    adjacency = tf.sparse.SparseTensor(indices, values, dense_shape=(N, N))
    
    # Imitation MLP
    mlp_output = tf.random.normal((N, K))
    assignments = tf.nn.softmax(mlp_output, axis=1)
    
    # Test slow
    start_time = tf.timestamp()
    mod_value = Metrics.modularity(adjacency, assignments)
    compute_time = tf.timestamp() - start_time
    print(f"\nComputation time: {compute_time.numpy():.6f} sec")
    print(f"Modularity value: {mod_value.numpy():.4f}")

    # Test fast
    start_time = tf.timestamp()
    mod_value = Metrics.fast_modularity(adjacency, assignments)
    compute_time = tf.timestamp() - start_time
    print(f"\nComputation time: {compute_time.numpy():.6f} sec")
    print(f"Modularity value: {mod_value.numpy():.4f}")
