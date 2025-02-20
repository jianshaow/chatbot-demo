#!/bin/bash

base_image=jianshao/trfs-dev-base
if [ "$trfs_ver" == "" ]; then
    trfs_ver=$(docker run --rm ${base_image}:latest pip list | grep transformers | awk '{print $2}')
else
    docker pull ${base_image}:${trfs_ver}
fi
echo "Using transformers version ${trfs_ver}"

image=jianshao/transformers-demo
docker build -t ${image}:${trfs_ver} . --build-arg TAG=${trfs_ver}

docker push ${image}:${trfs_ver}

echo "Done"
