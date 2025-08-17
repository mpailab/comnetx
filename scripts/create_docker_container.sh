#!/usr/bin/bash

NAME=""
DOCKER=docker
IMAGE="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"
REUSE=0
SHM_SIZE="2g"

function usage {
    echo "usage: $0 [-rmh] -n NAME [-m SHM_SIZE]"
    echo "  -n   Container's name"
    echo "  -r   Remove container with the same name if it exists"
    echo "  -m   Shared memory size for container (default: 2g)"
    echo "  -h   Display help"
    exit 1
}

[ $# -eq 0 ] && usage

PARSED_ARGUMENTS=$(getopt -n $0 -o n:m:rh -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
    usage
fi

# echo "PARSED_ARGUMENTS is $PARSED_ARGUMENTS"
eval set -- "$PARSED_ARGUMENTS"
while :
do
    case "$1" in
        -n)
            if [ "$NAME" != "" ]
            then
                echo "Conflict names: $NAME and $2 - this should not happen."
                usage
            fi
            NAME="${USER}_$2"
            shift 2
            ;;
        -r)
            REUSE=1
            shift
            ;;
        -m)
            SHM_SIZE="$2"
            shift 2
            ;;
        -h)
            usage
            ;;
        --) 
            shift
            break
            ;;
        *)
            echo "Unexpected option: $1 - this should not happen."
            usage
            ;;
    esac
done

if [ $# -ne 0 ]
then
    echo "Unexpected arguments: $@ - this should not happen."
    usage
fi

echo "$DOCKER"
if [ $REUSE -eq 1 ]
then
    printf "  stop "
    $DOCKER stop $NAME
    printf "  remove "
    $DOCKER rm $NAME
fi 

printf "  create $NAME as "
$DOCKER create --gpus all -it --shm-size=$SHM_SIZE \
    -e TERM=xterm-256color \
    -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH \
    --entrypoint /bin/bash \
    -v /auto/datasets/graphs:/auto/datasets/graphs \
    -v /home/$USER:/home/$USER \
    -v /home/$USER/.bashrc:/root/.bashrc \
    -w / \
    --name $NAME -h $NAME $IMAGE

printf "  start "
$DOCKER start $NAME

SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[INFO] Installing packages in container..."
$DOCKER exec -w "$PROJECT_ROOT" "$NAME" bash ./scripts/install_packages.sh
