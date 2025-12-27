# ComNetX
Community Detection Using Neural Networks

# Quick start

Create new docker container:
```shell
$  ./scripts/create_docker_container.sh -v -n <my_container_name>
``` 

Install the necessary packages inside the created container <my_container_name>:
```shell
root@user_<my_container_name>$ bash ./scripts/install_packages.sh
```

# Running on MGU‑270 (SLURM)

The commands below show how to start an interactive session on the MGU‑270 cluster for your own user account.
Before running them, replace:

<USER> – with your login on the cluster

all paths under /scratch/<USER>/... – with your actual image and dataset locations.

Open a new tmux session:
```shell
tmux new -s comnetx
```

Inside this tmux session, start an interactive container with srun:
```shell
srun --nodes=1 \
     --gpus=1 \
     --cpus-per-task=128 \
     --container-image /scratch/drobyshevas/diaduskaau_comnetx2_latest.sqsh \
     --container-workdir /scratch/<USER>/comnetx \
     --container-mounts=/scratch/drobyshevas/comnetx:/scratch/drobyshevas/comnetx, \
     /scratch/bokovgv/datasets/graphs/konect:/auto/datasets/graphs/dynamic_konect_project_datasets/, \
     /scratch/bokovgv/datasets/graphs/small:/auto/datasets/graphs/small  \
      --pty bash
```