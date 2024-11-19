ARG BASE_IMAGE=jianshao/torch-dev-base
ARG TAG=2.5.1

FROM ${BASE_IMAGE}:${TAG}

USER root
COPY packages.txt /
RUN /apt-install.sh /packages.txt
USER devel

COPY requirements*.txt ./
ARG REQUIREMENTS=requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${REQUIREMENTS}
