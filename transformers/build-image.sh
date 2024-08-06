#!/bin/bash

if [ "$torch_ver" == "" ]; then
    torch_ver=2.3.1
fi
echo "Using torch version: ${torch_ver}"

docker build -t jianshao/transformers-demo:latest . \
       --build-arg TAG=$torch_ver

transformers_ver=$(docker run --rm jianshao/transformers-demo:latest pip list | grep transformers | awk '{print $2}')
echo "Using transformers version ${transformers_ver}"

docker tag jianshao/transformers-demo:latest jianshao/transformers-demo:${transformers_ver}
docker push jianshao/transformers-demo:latest
docker push jianshao/transformers-demo:${transformers_ver}

echo "Done"
