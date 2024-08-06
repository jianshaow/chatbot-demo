#!/bin/bash

if [ "$torch_ver" == "" ]; then
    torch_ver=2.3.1
    echo "Using default torch version: ${torch_ver}"
fi

docker build -t jianshao/llamaindex-demo:latest . \
       --build-arg TAG=$torch_ver-gpu

llamaindex_ver=$(docker run --rm jianshao/llamaindex-demo:latest pip list | grep llama-index-core|awk '{print $2}')
echo "Using llama-index version ${llamaindex_ver}"

docker tag jianshao/llamaindex-demo:latest jianshao/llamaindex-demo:${llamaindex_ver}
docker push -a jianshao/llamaindex-demo
