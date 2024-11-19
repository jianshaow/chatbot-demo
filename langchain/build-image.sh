#!/bin/bash

if [ "$torch_ver" == "" ]; then
    torch_ver=2.5.1
fi
echo "Using torch version: ${torch_ver}"

base_image=jianshao/torch-dev-base
docker pull ${base_image}:${torch_ver}

image=jianshao/langchain-demo
docker build -t ${image}:latest . --build-arg TAG=$torch_ver $*

langchain_core_ver=$(docker run --rm ${image}:latest pip list | grep langchain-core | awk '{print $2}')
echo "Using langchain-core version ${langchain_core_ver}"

docker tag ${image}:latest ${image}:${langchain_core_ver}
docker push ${image}:latest
docker push ${image}:${langchain_core_ver}

echo "Done"
