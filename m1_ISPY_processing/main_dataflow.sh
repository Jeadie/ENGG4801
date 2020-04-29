#!/usr/bin/env bash


# Runs main pipeline on Google Cloud Dataflow
PROJECT="long-loop-273905"
REGION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"

#TODO: upload csvs to GCS 
# Run command
python main.py  \
    --runner=DataflowRunner \
    --region $REGION \
    --studies-dir="gs://ispy_dataquery/dicoms/" \
    --tfrecord-name="gs://ispy_dataquery/result" \
    --num-shards=2 \
    --patient-clinical="gs://ispy_dataquery/clinical.csv" \
    --project="$PROJECT" \
    --patient-outcomes="gs://ispy_dataquery/outcome.csv" \
    --temp_location="gs://ispy_dataquery/temp/" \
