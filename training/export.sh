#!/usr/bin/env bash

./build.sh

docker save stoictrain | gzip -c > STOICTrain.tar.gz