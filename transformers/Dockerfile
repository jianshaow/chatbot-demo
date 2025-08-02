ARG BASE_IMAGE=jianshao/trfs-dev-base
ARG TAG=4.49.0

FROM ${BASE_IMAGE}:${TAG} AS base

FROM base AS build
USER root
ARG CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
ADD ${CUDA_KEYRING_URL} cuda-keyring_all.deb
RUN dpkg -i cuda-keyring_all.deb && apt update && \
    apt install -y --no-install-recommends cuda-nvcc-12-6
USER devel
RUN pip wheel --no-deps flash-attn

FROM base
COPY --from=build $HOME/*.whl /tmp/
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir /tmp/*.whl && \
    rm -f /tmp/*
