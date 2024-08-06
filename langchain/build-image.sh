#!/bin/bash

if [ "$torch_ver" == "" ]; then
    torch_ver=2.3.1
fi
echo "Using torch version: ${torch_ver}"

docker build -t jianshao/langchain-demo:latest . \
       --build-arg TAG=$torch_ver

langchain_core_ver=$(docker run --rm jianshao/langchain-demo:latest pip list | grep langchain-core| awk '{print $2}')
echo "Using langchain-core version ${langchain_core_ver}"

docker tag jianshao/langchain-demo:latest jianshao/langchain-demo:${langchain_core_ver}
docker push jianshao/langchain-demo:latest
docker push jianshao/langchain-demo:${langchain_core_ver}

echo "Done"
