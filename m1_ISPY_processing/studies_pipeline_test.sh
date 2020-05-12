#!/usr/bin/env bash

### Simple Test to ensure the studies pipeline process correctly. A small dataset is used and the outputs are manually inspected.
python test_studies.py --test \
    --runner=DirectRunner \
    --studies-dir="gs://ispy_dataquery/dicoms/" \
    --num-series=1 \
