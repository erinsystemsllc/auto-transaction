#!/bin/bash

set -eu -o pipefail

docker build -t $IMAGE_REPOSITORY:$IMAGE_TAG .
docker push $IMAGE_REPOSITORY:$IMAGE_TAG
