#!/bin/bash
#SBATCH --job-name=torch_cuda_multi
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --output=torch_cuda_multi_%j.out

srun --nodes=1 --ntasks=2 --cpus-per-task=32 --gpus=4 \
     --container-image=/scratch/drobyshevas/diaduskaau_comnetx_latest.sqsh \
     --container-workdir=/scratch/drobyshevas/comnetx \
     --container-mounts=/scratch/drobyshevas/comnetx:/scratch/drobyshevas/comnetx,\
/scratch/bokovgv/datasets/graphs/konect:/auto/datasets/graphs/dynamic_konect_project_datasets/,\
/scratch/bokovgv/datasets/graphs/small:/auto/datasets/graphs/small \
     bash -lc '
  echo "task $SLURM_PROCID on $(hostname), SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

  case $SLURM_PROCID in
    0)
      export CUDA_VISIBLE_DEVICES=0,1
      TAG="taskA"
      ;;
    1)
      export CUDA_VISIBLE_DEVICES=2,3
      TAG="taskB"
      ;;
  esac

  python3 -c "import torch, os;
print(\"'$TAG' CUDA_VISIBLE_DEVICES=\", os.environ.get(\"CUDA_VISIBLE_DEVICES\"), \"cuda_count=\", torch.cuda.device_count());
[print(\"'$TAG' dev\", i, \"->\", torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

  python3 /scratch/drobyshevas/comnetx/scripts/show_gpu.py
'
