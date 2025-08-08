#!/usr/bin/bash

NAME=""
DOCKER=docker
IMAGE="pytorch/pytorch:2.1.0-cpu"
REUSE=0

function usage {
    echo "usage: $0 [-rvh] -n NAME"
    echo "  -n   Container's name"
    echo "  -r   Remove container with the same name if it exists"
    echo "  -v   Use nvidia-docker instead of docker"
    echo "  -h   Display help"
    exit 1
}

[ $# -eq 0 ] && usage

PARSED_ARGUMENTS=$(getopt -n $0 -o n:d:p:rvh -- "$@")
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
        -v)
            IMAGE="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"
            shift
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
$DOCKER create --gpus all -it --shm-size=2g \
    -e TERM=xterm-256color \
    --entrypoint /bin/bash \
    -v /auto/datasets/graphs:/auto/datasets/graphs \
    -w / \
    -v /home/$USER:/home/$USER \
    -v /home/$USER/.bashrc:/root/.bashrc --name $NAME -h $NAME $IMAGE

printf "  start "
$DOCKER start $NAME