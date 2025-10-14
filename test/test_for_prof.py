import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import time

def tensorflow_matrix_operations():
    """Операции с матрицами в TensorFlow"""
    # print("TensorFlow операции:")
    
    # Создание больших матриц
    size = 2000
    matrix_a = tf.random.normal([size, size], dtype=tf.float32)
    matrix_b = tf.random.normal([size, size], dtype=tf.float32)
    
    # Умножение матриц
    start_time = time.time()
    multiplication_result = tf.matmul(matrix_a, matrix_b)
    multiplication_time = time.time() - start_time
    
    # Сложение матриц
    start_time = time.time()
    addition_result = tf.add(matrix_a, matrix_b)
    addition_time = time.time() - start_time
    
    # Поэлементное умножение
    start_time = time.time()
    elementwise_result = tf.multiply(matrix_a, matrix_b)
    elementwise_time = time.time() - start_time
    
    # print(f"Умножение матриц: {multiplication_time:.4f} сек")
    # print(f"Сложение матриц: {addition_time:.4f} сек")
    # print(f"Поэлементное умножение: {elementwise_time:.4f} сек")
    
    return multiplication_result, addition_result, elementwise_result

def pytorch_matrix_operations():
    """Операции с матрицами в PyTorch"""
    # print("\nPyTorch операции:")
    
    # Проверка доступности GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Используемое устройство: {device}")
    
    # Создание больших матриц
    size = 2000
    matrix_a = torch.randn(size, size, device=device, dtype=torch.float32)
    matrix_b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # Умножение матриц
    start_time = time.time()
    multiplication_result = torch.mm(matrix_a, matrix_b)
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Ожидание завершения вычислений на GPU
    multiplication_time = time.time() - start_time
    
    # Сложение матриц
    start_time = time.time()
    addition_result = torch.add(matrix_a, matrix_b)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    addition_time = time.time() - start_time
    
    # Поэлементное умножение
    start_time = time.time()
    elementwise_result = torch.mul(matrix_a, matrix_b)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elementwise_time = time.time() - start_time
    
    # print(f"Умножение матриц: {multiplication_time:.4f} сек")
    # print(f"Сложение матриц: {addition_time:.4f} сек")
    # print(f"Поэлементное умножение: {elementwise_time:.4f} сек")
    
    return multiplication_result, addition_result, elementwise_result

def advanced_tensorflow_operations():
    """Продвинутые операции в TensorFlow"""
    # print("\nПродвинутые TensorFlow операции:")
    
    size = 1000
    batch_size = 10
    
    # Пакетные операции
    batch_matrices_a = tf.random.normal([batch_size, size, size], dtype=tf.float32)
    batch_matrices_b = tf.random.normal([batch_size, size, size], dtype=tf.float32)
    
    # Пакетное умножение матриц
    start_time = time.time()
    batch_multiplication = tf.matmul(batch_matrices_a, batch_matrices_b)
    batch_time = time.time() - start_time
    
    # Транспонирование
    start_time = time.time()
    transposed = tf.transpose(batch_matrices_a)
    transpose_time = time.time() - start_time
    
    # print(f"Пакетное умножение ({batch_size} матриц): {batch_time:.4f} сек")
    # print(f"Транспонирование: {transpose_time:.4f} сек")
    
    return batch_multiplication, transposed

def advanced_pytorch_operations():
    """Продвинутые операции в PyTorch"""
    # print("\nПродвинутые PyTorch операции:")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size = 1000
    batch_size = 10
    
    # Пакетные операции
    batch_matrices_a = torch.randn(batch_size, size, size, device=device)
    batch_matrices_b = torch.randn(batch_size, size, size, device=device)
    
    # Пакетное умножение матриц
    start_time = time.time()
    batch_multiplication = torch.bmm(batch_matrices_a, batch_matrices_b)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    batch_time = time.time() - start_time
    
    # Транспонирование
    start_time = time.time()
    transposed = batch_matrices_a.transpose(1, 2)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    transpose_time = time.time() - start_time
    
    # print(f"Пакетное умножение ({batch_size} матриц): {batch_time:.4f} сек")
    # print(f"Транспонирование: {transpose_time:.4f} сек")
    
    return batch_multiplication, transposed

def matrix_decomposition_operations():
    """Операции матричного разложения"""
    # print("\nОперации матричного разложения:")
    
    size = 500
    
    # TensorFlow SVD
    tf_matrix = tf.random.normal([size, size], dtype=tf.float32)
    start_time = time.time()
    s, u, v = tf.linalg.svd(tf_matrix)
    tf_svd_time = time.time() - start_time
    
    # PyTorch SVD
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_matrix = torch.randn(size, size, device=device)
    start_time = time.time()
    u_torch, s_torch, v_torch = torch.svd(torch_matrix)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    torch_svd_time = time.time() - start_time
    
    # print(f"TensorFlow SVD: {tf_svd_time:.4f} сек")
    # print(f"PyTorch SVD: {torch_svd_time:.4f} сек")

def main():
    """Основная функция"""
    # print("Операции с большими матрицами в TensorFlow и PyTorch")
    # print("=" * 60)
    
    # Базовые операции
    tf_mult, tf_add, tf_elem = tensorflow_matrix_operations()
    torch_mult, torch_add, torch_elem = pytorch_matrix_operations()
    
    # Продвинутые операции
    tf_batch, tf_trans = advanced_tensorflow_operations()
    torch_batch, torch_trans = advanced_pytorch_operations()
    
    # Операции разложения
    matrix_decomposition_operations()
    
    # print("\nВсе операции завершены успешно!")

if __name__ == "__main__":
    main()