ARG BASE_IMAGE=jianshao/trfs-dev-base
ARG TAG=4.49.0

FROM ${BASE_IMAGE}:${TAG}

RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
