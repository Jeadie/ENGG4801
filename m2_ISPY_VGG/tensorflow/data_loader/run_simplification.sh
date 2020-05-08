if [[ "$GOOGLE_APPLICATION_CREDENTIALS" == "" ]]; then 
  echo "Please export GOOGLE_APPLICATION_CREDENTIALS env variable."
  exit 1
fi
gcloud config configurations activate maxwell
DOCKER_IMAGE_NAME=gcr.io/long-loop-273905/m2_simplify:latest


docker build . -f Dockerfile -t $DOCKER_IMAGE_NAME
docker push $DOCKER_IMAGE_NAME
kubectl apply -f job.yaml

