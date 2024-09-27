#!/bin/bash

if [ "$torch_ver" == "" ]; then
    torch_ver=2.4.1
fi
echo "Using torch version: ${torch_ver}"

image=jianshao/transformers-demo
docker build -t ${image}:latest . --build-arg TAG=$torch_ver

transformers_ver=$(docker run --rm ${image}:latest pip list | grep transformers | awk '{print $2}')
echo "Using transformers version ${transformers_ver}"

docker tag ${image}:latest ${image}:${transformers_ver}
docker push ${image}:latest
docker push ${image}:${transformers_ver}

echo "Done"
