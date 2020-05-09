##
# Sets the gcloud configuration or creates a thesis specific config for the long-loop project.
##

gcloud config configurations activate thesis-gcp
if [ $? -ne  0 ]; then
  gcloud config set project long-loop-273905
  gcloud config set account data-processing@long-loop-273905.iam.gserviceaccount.com 
  gcloud config configurations create thesis-gcp
 gcloud config configurations activate thesis-gcp
fi 

if [[ -n $GOOGLE_APPLICATION_CREDENTIALS ]]; then
  gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
else
  echo "Please set GOOGLE_APPLICATION_CREDENTIALS, or run gcloud auth activate-service-account --key-file <Google Credentials>"
fi 

