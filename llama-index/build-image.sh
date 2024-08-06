#!/bin/bash

if [ "$torch_ver" == "" ]; then
    torch_ver=2.3.1
fi
echo "Using torch version: ${torch_ver}"

docker build -t jianshao/llamaindex-demo:latest . \
       --build-arg TAG=$torch_ver

llamaindex_ver=$(docker run --rm jianshao/llamaindex-demo:latest pip list | grep llama-index-core| awk '{print $2}')
echo "Using llama-index version ${llamaindex_ver}"

docker tag jianshao/llamaindex-demo:latest jianshao/llamaindex-demo:${llamaindex_ver}
docker push jianshao/llamaindex-demo:latest
docker push jianshao/llamaindex-demo:${llamaindex_ver}

echo "Done"
