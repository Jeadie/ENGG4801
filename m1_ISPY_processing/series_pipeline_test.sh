#!/usr/bin/env bash

### Simple Test to ensure the series pipeline process correctly. A small dataset is used and the outputs are manually inspected.
python series_pipeline.py --test \
    --runner=DirectRunner \
    --studies-dir="./studies/" \#"gs://ispy_dataquery/dicoms/" \
    --num-series=1\
    --patient-clinical="clinical_small.csv" \
    --patient-outcomes="outcome_small.csv" \
