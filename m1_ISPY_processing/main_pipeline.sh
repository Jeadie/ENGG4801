#!/usr/bin/env bash

### Simple Test to ensure the studies pipeline process correctly. A small dataset is used and the outputs are manually inspected.
python main.py  \
    --runner=DirectRunner \
    --num-series=2\
    --studies-dir="gs://ispy_dataquery/dicoms/" \ "./studies/" \
    --tfrecord-name="./result" \
    --num-shards=10 \
