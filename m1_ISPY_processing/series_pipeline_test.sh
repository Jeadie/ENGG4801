#!/usr/bin/env bash

# Runs main pipeline on Google Cloud Dataflow
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"

# Run command
python test_series.py --test \
    --runner=DirectRunner \
    --num-series=1 \
    --studies-dir="gs://ispy_dataquery/dicoms/" \
    --tfrecord-name="name" \
    --num-shards=2 \
