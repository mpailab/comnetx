# import torch
# import torch.profiler as profiler

# # Базовая модель и данные
# model = torch.nn.Linear(100, 10).cuda()
# x = torch.randn(64, 100).cuda()

# # Профилирование одного forward pass
# with profiler.profile(
#     activities=[
#         profiler.ProfilerActivity.CPU,
#         profiler.ProfilerActivity.CUDA,  # Для GPU
#     ]
# ) as prof:
#     output = model(x)
#     loss = output.sum()
#     loss.backward()

# # Вывод результатов в консоль
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

import torch
import torch.profiler as profiler

# Создаем матрицы разных размеров
matrix1 = torch.randn(2048, 4096).cuda()
matrix2 = torch.randn(4096, 1024).cuda()
matrix3 = torch.randn(1024, 2048).cuda()

# Профилирование различных операций с матрицами
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    # record_shapes=True,
    profile_memory=True,
    with_modules=True
) as prof:
    # Цепочка перемножений матриц
    temp1 = torch.matmul(matrix1, matrix2)    # 2048x4096 * 4096x1024 = 2048x1024
    temp2 = torch.matmul(temp1, matrix3)      # 2048x1024 * 1024x2048 = 2048x2048
    temp3 = torch.matmul(temp2, temp2.T)      # 2048x2048 * 2048x2048 = 2048x2048
    
    # Элементные операции
    temp4 = temp3 * 2.0                       # Умножение на скаляр
    temp5 = temp4 + temp3                     # Сложение матриц
    
    # Batch matrix multiplication
    batch_mat1 = torch.randn(32, 256, 256).cuda()
    batch_mat2 = torch.randn(32, 256, 256).cuda()
    batch_result = torch.bmm(batch_mat1, batch_mat2)  # Пакетное перемножение
   
    # Финальный результат
    final_result = temp5.sum() + batch_result.sum()

# Вывод результатов
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1))