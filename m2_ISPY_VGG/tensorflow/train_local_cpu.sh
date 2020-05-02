#!/usr/bin/env bash
##########################################################
# where to write tfevents
OUTPUT_DIR="gs://data"
# experiment settings
TRAIN_BATCH=10
EVAL_BATCH=10
LR=0.002
EPOCHS=1
# create a job name for the this run
prefix="example"
now=$(date +"%Y%m%d_%H_%M_%S")
JOB_NAME="$ENV_NAME"-"$prefix"_"$now"
# locations locally or on the cloud for your files
TRAIN_FILES="data/result.tfrecords"
EVAL_FILES="data/result.tfrecords"
TEST_FILES="data/result.tfrecords"
##########################################################

# get current working directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# create folders if they don't exist of logs and outputs
mkdir -p ${DIR}/runlogs

# create a local job directory for checkpoints etc
JOB_DIR=${OUTPUT_DIR}/${JOB_NAME}

export CUDA_VISIBLE_DEVICES=""
# start training
python3 -m initialisers.task \
        --job-dir ${JOB_DIR} \
        --train-batch-size ${TRAIN_BATCH} \
        --eval-batch-size ${EVAL_BATCH} \
        --learning-rate ${LR} \
        --num-epochs ${EPOCHS} \
        --train-files ${TRAIN_FILES} \
        --eval-files ${EVAL_FILES} \
        --test-files ${TEST_FILES} \
        --export-path "${OUTPUT_DIR}exports" \

echo "Job launched."
