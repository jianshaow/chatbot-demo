#!/bin/bash

if [ "$transformers_ver" == "" ]; then
    transformers_ver=latest
fi
echo "Using transformers version: ${transformers_ver}"

base_image=jianshao/transformers-demo
docker pull ${base_image}:${transformers_ver}

image=jianshao/llamaindex-demo
docker build -t ${image}:latest . --build-arg TAG=${transformers_ver} $*

llamaindex_ver=$(docker run --rm ${image}:latest pip list | grep llama-index-core | awk '{print $2}')
echo "Using llama-index version ${llamaindex_ver}"

docker tag ${image}:latest ${image}:${llamaindex_ver}
docker push ${image}:latest
docker push ${image}:${llamaindex_ver}

echo "Done"
