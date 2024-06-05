ARG BASE_IMAGE=jianshao/torch-dev-base
ARG TAG=2.3.0-cpu

FROM ${BASE_IMAGE}:${TAG}

COPY requirements*.txt ./
ARG REQUIREMENTS=requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${REQUIREMENTS}
