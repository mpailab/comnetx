#!/usr/bin/bash

NAME=""
DOCKER=docker
IMAGE="diaduskaau/comnetx:latest"
REUSE=0
SHM_SIZE="2g"
GPUS="all"


SCRIPT_DIR=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
PROJECT_USER=$(basename "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")
PROJECT_DIR="/home/dev/users/$PROJECT_USER/comnetx"


SERVER=$(hostname)


function usage {
    echo "usage: $0 [-rmh] -n NAME [-m SHM_SIZE]"
    echo "  -n   Container's name"
    echo "  -r   Remove container with the same name if it exists"
    echo "  -m   Shared memory size for container (default: 2g)"
    echo "  -g   GPUs to use (default: all, example: 0,1,2 или 7)"
    echo "  -h   Display help"
    exit 1
}


[ $# -eq 0 ] && usage


PARSED_ARGUMENTS=$(getopt -n $0 -o n:m:rhg: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
    usage
fi


eval set -- "$PARSED_ARGUMENTS"
while :
do
    case "$1" in
        -n)
            if [ "$NAME" != "" ]; then
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
        -g)
            GPUS="$2"
            shift 2
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


if [ $# -ne 0 ]; then
    echo "Unexpected arguments: $@ - this should not happen."
    usage
fi


if $DOCKER ps -a --format '{{.Names}}' | grep -q "^$NAME$"; then
    if [ $REUSE -eq 0 ]; then
        echo "[ERROR] Container '$NAME' already exists. Use -r to recreate."
        exit 1
    fi
fi


echo "$DOCKER"
if [ $REUSE -eq 1 ]; then
    printf "  stop "
    $DOCKER stop $NAME || true
    printf "  remove "
    $DOCKER rm $NAME || true
fi 


# Убрал блок выбора пути в зависимости от сервера (astra/cn69)


echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "PROJECT_USER: $PROJECT_USER"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "SERVER: $SERVER"


printf "  create $NAME as "
$DOCKER create --gpus "device=$GPUS" -it --shm-size=$SHM_SIZE \
    -e TERM=xterm-256color \
    --entrypoint /bin/bash \
    -w /home/dev/users/$PROJECT_USER/comnetx \
    -v /home/$USER:/home/$USER \
    -v /home/$USER/.bashrc:/root/.bashrc --name $NAME -h $NAME $IMAGE


printf "  start "
$DOCKER start $NAME

$DOCKER exec -it $NAME /bin/bash


read -p "Save changes to image $IMAGE? (y/N): " SAVE
if [[ "$SAVE" =~ ^[Yy]$ ]]; then
    echo "[INFO] Commmiting changes..."
    $DOCKER commit $NAME $IMAGE


    read -p "Push updated image to Docker Hub? (y/N): " PUSH
    if [[ "$PUSH" =~ ^[Yy]$ ]]; then
        echo "[INFO] Pushing image to Docker Hub..."
        $DOCKER push $IMAGE
    fi
fi


echo "[INFO] Done."