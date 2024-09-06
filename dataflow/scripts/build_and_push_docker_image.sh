#!/bin/bash

# Run the following command beforehand:
# gcloud auth login
# gcloud auth configure-docker

set -eux

source scripts/gcp_setup.sh

# export python dependencies as requirements.txt
poetry export -f requirements.txt --output requirements.txt

# build docker image
docker build -t $IMAGE_URI .
docker push $IMAGE_URI

# delete old revisions
docker image prune -f
gcloud container images list-tags $IMAGE_URI --filter='-tags:*' --format="get(digest)" | while read -r digest; do
    gcloud container images delete -q "$IMAGE_URI@$digest"
done
