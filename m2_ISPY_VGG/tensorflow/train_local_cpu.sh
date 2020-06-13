#!/usr/bin/env bash

# Only set default variables if running locally. Else use those from job or environment.
if [ "$1" != "--job" ]; then
    # where to write tfevents
    OUTPUT_DIR="runlogs/"
    # experiment settings
    TRAIN_BATCH=3
    EVAL_BATCH=3
    LEARNING_RATE=0.002
    EPOCHS=1

    prefix="$1"
    
    JOB_NAME="${prefix}_$(date +'%Y%m%d_%H_%M_%S')"
    JOB_DIR="$(pwd)/${OUTPUT_DIR}"

    # locations locally or on the cloud for your files
    TRAIN_FILES="output/combined_total.tfrecords"  
    EVAL_FILES="output/"
    TEST_FILES="output/"
    DATA_LOADER="ResampledTFRecord" # TFRecordShardLoader
fi

if [ "$GCS_FUSE_BUCKET" != "" ]; then
    # Mount storage bucket to folder
    mkdir $GCS_FUSE_BUCKET
    gcsfuse $GCS_FUSE_BUCKET "$(pwd)/$GCS_FUSE_BUCKET/"
    $JOB_DIR="$GCS_FUSE_BUCKET/${OUTPUT_DIR}"
fi 

python3 -m initialisers.task \
        --job-dir ${JOB_DIR} \
        --job-name ${JOB_NAME} \
        --data-loader ${DATA_LOADER} \
        --train-batch-size ${TRAIN_BATCH} \
        --eval-batch-size ${EVAL_BATCH} \
        --learning-rate ${LEARNING_RATE} \
        --num-epochs ${EPOCHS} \
        --train-files ${TRAIN_FILES} \
        --eval-files ${EVAL_FILES} \
        --test-files ${TEST_FILES} \
        --use-stack 0 \
        --export-path "${OUTPUT_DIR}/exports" \


