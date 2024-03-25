#!/bin/bash

if [ "$*" != "" ]; then
  $*
  exit 0
fi

if [ "$MODEL_PATH" == "" ]; then
  MODEL_PATH="TheBloke/vicuna-13B-v1.5-AWQ"
fi

python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH
