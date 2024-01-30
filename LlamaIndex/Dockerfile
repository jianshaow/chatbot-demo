ARG BASE_IMAGE=jianshao/dl-runtime-base
ARG TAG=3.11-slim

FROM ${BASE_IMAGE}:${TAG}

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
