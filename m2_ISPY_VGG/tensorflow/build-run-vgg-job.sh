if [[ "$GOOGLE_APPLICATION_CREDENTIALS" == "" ]]; then 
  echo "Please export GOOGLE_APPLICATION_CREDENTIALS env variable."
  exit 1
fi

gcloud config configurations activate maxwell
DOCKER_IMAGE_NAME=gcr.io/long-loop-273905/m2_vgg:latest


docker build . -f Dockerfile -t $DOCKER_IMAGE_NAME
if [[ $? -eq 1  ]]; then
    echo "[BUILD] - Failed to build Docker Image."
    exit 1
fi

docker push $DOCKER_IMAGE_NAME
if [[ $? -eq 1  ]]; then
    echo "[BUILD] - Failed to push Docker Image: $DOCKER_IMAGE_NAME."
    exit 1
fi

kubectl apply -f vgg-job.yaml
