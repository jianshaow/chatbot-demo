ARG BASE_IMAGE=jianshao/torch-dev-base
ARG TAG=2.4.1

FROM ${BASE_IMAGE}:${TAG} as build
USER root
ADD https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb cuda-keyring_all.deb
RUN dpkg -i cuda-keyring_all.deb && apt update && \
    apt install -y --no-install-recommends cuda-nvcc-12-4
USER devel
RUN pip install --no-cache-dir flash-attn

FROM ${BASE_IMAGE}:${TAG}
COPY --from=build /home/devel/.local/lib/python3.12/site-packages/flash_attn/ /home/devel/.local/lib/python3.12/site-packages/flash_attn
COPY --from=build /home/devel/.local/lib/python3.12/site-packages/flash_attn-*.dist-info/ /home/devel/.local/lib/python3.12/site-packages/flash_attn-*.dist-info/
COPY --from=build /home/devel/.local/lib/python3.12/site-packages/flash_attn*.so /home/devel/.local/lib/python3.12/site-packages/
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
