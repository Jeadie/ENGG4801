#!/usr/bin/env bash


# Runs main pipeline on Google Cloud Dataflow
PROJECT="long-loop-273905"
REGION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"

#TODO: upload csvs to GCS 
# Run command
python main.py  \
    --runner=DataflowRunner \
    --region=$REGION \
    --num_workers=2 \
    --max_num_workers=4 \
    --num-series=4 \
    --studies-dir="gs://ispy_dataquery/dicoms/" \
    --tfrecord-name="gs://ispy_dataquery/result_specific_desc" \
    --num-shards=2 \
    --series-descriptions="gs://ispy_dataquery/ISPY1_MetaData.csv" \
    --project "$PROJECT" \
    --temp_location="gs://ispy_dataquery/temp/" \
