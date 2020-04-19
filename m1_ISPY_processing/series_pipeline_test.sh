#/bin/bash

### Simple Test to ensure the patient pipeline process correctly. A small dataset is used and the outputs are manually inspected.
python series_pipeline.py --test --runner=DirectRunner --studies-dir="gs://ispy_dataquery/dicoms/" --num-series=5 --patient-clinical="clinical_small.csv" --patient-outcomes="outcome_small.csv"
