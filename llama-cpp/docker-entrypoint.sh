#!/bin/bash

if [ "$HF_REPO_ID" == "" ]; then
  HF_REPO_ID="TheBloke/vicuna-13B-v1.5-GGUF"
fi

if [ "${HF_MODEL_FILE}" == "" ]; then
  HF_MODEL_FILE="vicuna-13b-v1.5.Q4_K_M.gguf"
fi

model_path = $(huggingface-cli download $HF_REPO_ID $HF_MODEL_FILE)

python -m llama_cpp.server --host 0.0.0.0 --chat_format chatml --model $model_path
