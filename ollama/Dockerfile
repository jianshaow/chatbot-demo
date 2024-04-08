ARG BASE_IMAGE=ollama/ollama
ARG TAG=0.1.31

FROM ${BASE_IMAGE}:${TAG}

RUN useradd -ms /bin/bash devel

USER devel
ENV HOME=/home/devel
ENV PATH=$HOME/.local/bin:$PATH

WORKDIR $HOME
