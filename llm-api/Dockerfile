ARG BASE_IMAGE=jianshao/dl-dev-base
ARG TAG=3.11

FROM ${BASE_IMAGE}:${TAG}

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
