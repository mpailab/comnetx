#!/bin/bash
#SBATCH --job-name=leiden_multi
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --output=leiden_multi.out

srun --ntasks=3 --cpus-per-task=32 --cpu-bind=cores \
     --container-image=/scratch/konovalovay/dyn_graphs_cpp.sqsh \
     --container-workdir=/dynamic-graphs \
     --container-mounts=/scratch/konovalovay/dynamic-graphs:/dynamic-graphs,/scratch/bokovgv/datasets/graphs/konect:/dynamic-graphs/datasets \
     bash -lc '
       case $SLURM_PROCID in
         0) CONFIG="leidenalg_j1_b1000_asia" ;;
         1) CONFIG="leidenalg_j1_b1000_com-lj" ;;
         2) CONFIG="leidenalg_j1_b1000_com-orkut" ;;
       esac
       CMD="python3 scripts/launch.py conf/launch/msu/b1000/${CONFIG}.json &> logs/${USER}_${CONFIG}.log"
       echo "task $SLURM_PROCID on $(hostname), CONFIG=$CONFIG"
       eval "$CMD"
     '
