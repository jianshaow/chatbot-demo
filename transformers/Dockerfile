ARG BASE_IMAGE=jianshao/trfs-dev-base
ARG TAG=4.49.0
ARG FLASH_ATTN_VER=2.7.4.post1

FROM ${BASE_IMAGE}:${TAG} as base
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM base as build
USER root
ARG CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
ADD ${CUDA_KEYRING_URL} cuda-keyring_all.deb
RUN dpkg -i cuda-keyring_all.deb && apt update && \
    apt install -y --no-install-recommends cuda-nvcc-12-4
USER devel
RUN pip install --no-cache-dir flash-attn==${FLASH_ATTN_VER}

FROM base
ARG LOCAL_PY_PKG_PATH=/home/devel/.local/lib/python3.12/site-packages
COPY --from=build ${LOCAL_PY_PKG_PATH}/flash_attn ${LOCAL_PY_PKG_PATH}/flash_attn
COPY --from=build ${LOCAL_PY_PKG_PATH}/flash_attn-${FLASH_ATTN_VER}.dist-info \
                  ${LOCAL_PY_PKG_PATH}/flash_attn-${FLASH_ATTN_VER}.dist-info
COPY --from=build ${LOCAL_PY_PKG_PATH}/flash_attn*.so ${LOCAL_PY_PKG_PATH}/
