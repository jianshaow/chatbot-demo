ARG BASE_IMAGE=jianshao/dl-rt-base
ARG TAG=3.11

FROM ${BASE_IMAGE}:${TAG}

COPY requirements.txt docker-entrypoint.sh ./

ARG LC_INDEX_ARG="https://abetlen.github.io/llama-cpp-python/whl/cpu"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir llama-cpp-python[server] -i ${LC_INDEX_ARG}

ENTRYPOINT ["./docker-entrypoint.sh"]
