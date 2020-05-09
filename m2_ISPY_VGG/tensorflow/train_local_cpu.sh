#!/usr/bin/env bash

# Only set default variables if running locally. Else use those from job or environment.
if [ "$1" != "--job" ]; then
    # where to write tfevents
    OUTPUT_DIR="data"
    # experiment settings
    TRAIN_BATCH=10
    EVAL_BATCH=10
    LEARNING_RATE=0.002
    EPOCHS=1

    prefix="example"
    now=$(date +"%Y%m%d_%H_%M_%S")
    JOB_NAME="$ENV_NAME"-"$prefix"_"$now"

    # locations locally or on the cloud for your files
    TRAIN_FILES="data_loader/new-boi_1.tfrecords"
    EVAL_FILES="data/new-boi_1.tfrecords"
    TEST_FILES="data/new-boi_1.tfrecords"
fi

if [ "$GCS_FUSE_BUCKET" != "" ]; then
    # Mount storage bucket to folder
    mkdir $GCS_FUSE_BUCKET
    gcsfuse $GCS_FUSE_BUCKET "$(pwd)/$GCS_FUSE_BUCKET/"
fi 

python3 -m initialisers.task \
        --job-dir "$(pwd)/$GCS_FUSE_BUCKET/${JOB_DIR}" \
        --train-batch-size ${TRAIN_BATCH} \
        --eval-batch-size ${EVAL_BATCH} \
        --learning-rate ${LEARNING_RATE} \
        --num-epochs ${EPOCHS} \
        --train-files ${TRAIN_FILES} \
        --eval-files ${EVAL_FILES} \
        --test-files ${TEST_FILES} \
        --export-path "${OUTPUT_DIR}/exports" \
