#!/bin/bash

if [ "$torch_ver" == "" ]; then
    torch_ver=2.5.1
fi
echo "Using torch version: ${torch_ver}"

base_image=jianshao/torch-dev-base
docker pull ${base_image}:${torch_ver}

image=jianshao/llamaindex-demo
docker build -t ${image}:latest . --build-arg TAG=${torch_ver} $*

llamaindex_ver=$(docker run --rm ${image}:latest pip list | grep llama-index-core | awk '{print $2}')
echo "Using llama-index version ${llamaindex_ver}"

docker tag ${image}:latest ${image}:${llamaindex_ver}
docker push ${image}:latest
docker push ${image}:${llamaindex_ver}

echo "Done"
