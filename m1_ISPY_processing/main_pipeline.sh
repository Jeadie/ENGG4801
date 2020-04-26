#!/usr/bin/env bash

### Simple Test to ensure the studies pipeline process correctly. A small dataset is used and the outputs are manually inspected.
python main.py  \
    --runner=DirectRunner \
    --studies-dir="./" \
    --num-series=2 \
    --tfrecord-name="./result" \
    --num-shards=1 \
    --patient-clinical="clinical.csv" \
    --patient-outcomes="outcome.csv" \
