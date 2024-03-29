ARG BASE_IMAGE=jianshao/torch-dev-base
ARG TAG=2.2.1-cpu

FROM ${BASE_IMAGE}:${TAG}

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
