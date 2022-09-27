#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

# make sure the artifact and scratch folders are writable
#chmod 777 $SCRIPTPATH/../inference/artifact/
#chmod 777 $2/

# Clear the artifact and scratch folder
#rm -r $SCRIPTPATH/../inference/artifact/*
#rm -r $2/*

# Run the algorithm
MEMORY="128g"

docker run --rm --gpus all\
        --memory $MEMORY --memory-swap $MEMORY \
        --cap-drop ALL --cap-add SYS_NICE --security-opt "no-new-privileges" \
        --network none --shm-size 32g --pids-limit 1024 \
        -v $1:/input/  \
        -v $SCRIPTPATH/../inference/artifact/:/output/ \
        -v $2:/scratch/ \
        stoictrain