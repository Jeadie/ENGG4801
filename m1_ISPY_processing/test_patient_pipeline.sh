### Simple Test to ensure the patient pipeline process correctly. A small dataset is used and the outputs are manually inspected.
python patient_pipeline.py --test --runner=DirectRunner --patient-clinical="clinical_small.csv" --patient-outcomes="outcome_small.csv"
