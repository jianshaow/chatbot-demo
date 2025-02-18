#!/bin/bash

if [ "$transformers_ver" == "" ]; then
    transformers_ver=latest
fi
echo "Using transformers version: ${transformers_ver}"

base_image=jianshao/transformers-demo
docker pull ${base_image}:${transformers_ver}

image=jianshao/langchain-demo
docker build -t ${image}:latest . --build-arg TAG=$transformers_ver $*

langchain_core_ver=$(docker run --rm ${image}:latest pip list | grep langchain-core | awk '{print $2}')
echo "Using langchain-core version ${langchain_core_ver}"

docker tag ${image}:latest ${image}:${langchain_core_ver}
docker push ${image}:latest
docker push ${image}:${langchain_core_ver}

echo "Done"
