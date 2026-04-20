#!/bin/bash

[ ! -d "/home/dev/users/drobyshev/comnetx/logs" ] && mkdir -p /home/dev/users/drobyshev/comnetx/logs

#НАСТРОЙКИ — МЕНЯЙТЕ ТОЛЬКО ЗДЕСЬ:
configs=("config1.yaml" "config2.yaml" "config3.yaml")
gpus=(0 1 2)

echo "Запуск параллельных тестов GPU..."

# ШАБЛОН КОМАНДЫ — ВСТАВЛЯЙТЕ СВОИ ЗДЕСЬ:
for i in ${!configs[@]}; do
    gpu=${gpus[$i]}
    config=${configs[$i]}
    log_file="/home/dev/users/drobyshev/comnetx/logs/log_gpu${gpu}.txt"
    
    CUDA_VISIBLE_DEVICES=$gpu python /home/dev/users/drobyshev/comnetx/scripts/show_gpu.py > "$log_file" 2>&1 &

    echo "Запущен тест GPU $gpu: $config (PID: $!)"
done

echo "Ждём завершения тестов..."
wait

echo "Все тесты завершены!"
echo "Логи:"
ls -lh /home/dev/users/drobyshev/comnetx/logs/log_gpu*.txt
cat /home/dev/users/drobyshev/comnetx/logs/log_gpu*.txt