#!/bin/bash

if [ "$*" != "" ]; then
  $*
  exit 0
fi

if [ "$HF_REPO_ID" == "" ]; then
  HF_REPO_ID="TheBloke/vicuna-13B-v1.5-GGUF"
fi

if [ "${HF_MODEL_FILE}" == "" ]; then
  HF_MODEL_FILE="vicuna-13b-v1.5.Q4_K_M.gguf"
fi

if [ "${HF_MODEL_ALIAS}" == "" ]; then
  HF_MODEL_ALIAS="vicuna-13B-v1.5"
fi

if [ "${CUDA_VERSION}" != "" ]; then
  GPU_ARGS="--n_gpu_layers -1"
fi

python3 -m llama_cpp.server --host 0.0.0.0 --chat_format chatml $GPU_ARGS $LC_ARGS \
        --hf_model_repo_id $HF_REPO_ID --model $HF_MODEL_FILE --model_alias $HF_MODEL_ALIAS
