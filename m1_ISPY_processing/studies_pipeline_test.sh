#!/usr/bin/env bash

### Simple Test to ensure the studies pipeline process correctly. A small dataset is used and the outputs are manually inspected.
python studies_pipeline.py --test \
    --runner=DirectRunner \
    --studies-dir="./studies/" \ #"gs://ispy_dataquery/dicoms/" \
    --num-series=2 \
    --patient-clinical="clinical_small.csv" \
    --patient-outcomes="outcome_small.csv" \
