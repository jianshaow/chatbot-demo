ARG BASE_IMAGE=jianshao/dl-rt-base
ARG TAG=3.11-slim

FROM ${BASE_IMAGE}:${TAG}

COPY requirements.txt get_model ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    mv get_model ${HOME}/.local/bin/
