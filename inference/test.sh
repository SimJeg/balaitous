#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker volume create stoicalgorithm-output

# Run the algorithm
docker run --rm  \
        --memory=16g \
        -v $SCRIPTPATH/test/:/input/ \
        -v stoicalgorithm-output:/output/ \
        stoicalgorithm

docker run --rm \
        -v stoicalgorithm-output:/output/ \
        python:3.7-slim cat /output/probability-covid-19.json | python -m json.tool

docker run --rm \
        -v stoicalgorithm-output:/output/ \
        python:3.7-slim cat /output/probability-severe-covid-19.json | python -m json.tool


docker run --rm --gpus all \
        -v stoicalgorithm-output:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.7-slim python -c "import json, sys; f1a = json.load(open('/output/probability-covid-19.json')); f2a = json.load(open('/input/expected_output-probability-covid-19.json')); f1b = json.load(open('/output/probability-severe-covid-19.json')); f2b = json.load(open('/input/expected_output-probability-severe-covid-19.json')); sys.exit((f1a != f2a)|(f1b != f2b));"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm stoicalgorithm-output
