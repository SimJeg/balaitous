#!/usr/bin/env bash

./build.sh

docker save stoicalgorithm | gzip -c > STOICAlgorithm.tar.gz