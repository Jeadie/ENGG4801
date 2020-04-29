#!/usr/bin/env bash

### Simple Test to ensure the studies pipeline process correctly. A small dataset is used and the outputs are manually inspected.
python main.py  \
    --runner=DirectRunner \
    --studies-dir="gs://ispy_dataquery/dicoms/" \ "./studies/" \
    --tfrecord-name="./result" \
    --num-shards=10 \
    --patient-clinical="clinical.csv" \
    --patient-outcomes="outcome.csv" \
