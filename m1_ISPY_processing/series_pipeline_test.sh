#!/usr/bin/env bash

# Runs main pipeline on Google Cloud Dataflow
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"
# Run command
python series_pipeline.py --test \
    --runner=DirectRunner \
    --num-series=1 \
    --studies-dir="gs://ispy_dataquery/dicom/" \
    --tfrecord-name="name" \
    --num-shards=2 \
    --patient-clinical="clinical.csv" \
    --series-descriptions="ISPY1_MetaData.csv" \
    --patient-outcomes="outcome.csv" \   
    --test \
