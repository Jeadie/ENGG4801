

# Runs main pipeline on Google Cloud Dataflow
PROJECT="long-loop-273905"
REGION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"

python test_patient.py --test --runner=DirectRunner

