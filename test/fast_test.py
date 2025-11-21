import tensorflow as tf
import numpy as np
import time

# Проверяем доступность GPU
print("Доступные устройства:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

# Создаем матрицы
matrix_a = tf.constant(np.random.rand(1000, 1000), dtype=tf.float32)
matrix_b = tf.constant(np.random.rand(1000, 1000), dtype=tf.float32)

# Перемножение матриц (автоматически использует GPU если доступно)
start_time = time.time()
result = tf.matmul(matrix_a, matrix_b)
end_time = time.time()

print(f"\nВремя выполнения: {end_time - start_time:.4f} секунд")
print(f"Размер результата: {result.shape}")
print(f"Устройство выполнения: {result.device}")