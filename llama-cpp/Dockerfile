ARG BASE_IMAGE=jianshao/dl-dev-base
ARG TAG=3.11

FROM ${BASE_IMAGE}:${TAG}

COPY apt-install.sh requirements.txt docker-entrypoint.sh ./

ARG PIP_INDEX_URL=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ARG CMAKE_ARGS=""
ENV CMAKE_ARGS=${CMAKE_ARGS}
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["./docker-entrypoint.sh"]
EXPOSE 8000
