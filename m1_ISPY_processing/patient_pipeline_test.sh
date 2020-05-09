

# Runs main pipeline on Google Cloud Dataflow
PROJECT="long-loop-273905"
REGION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"

#TODO: upload csvs to GCS 
# Run command
python patient_pipeline.py  \
    --test \
    --runner=DirectRunner \
    --patient-clinical="clinical.csv" \
    --patient-outcomes="outcome.csv" \
    --temp_location="gs://ispy_dataquery/temp/" \
