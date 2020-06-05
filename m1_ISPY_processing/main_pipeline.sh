#!/usr/bin/env bash
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"
### Simple Test to ensure the studies pipeline process correctly. A small dataset is used and the outputs are manually inspected.
python main.py  \
    --runner=DirectRunner \
    --batch-size=20 \
    --studies-dir="gs://ispy_dataquery/dicoms/" \ "./studies/" \
    --tfrecord-name="./data/result" \
    --num-shards=2 \
