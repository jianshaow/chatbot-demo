ARG BUILD_IMAGE=jianshao/dl-builder
ARG BUILD_TAG=3.11

ARG BASE_IMAGE=jianshao/dl-rt-base
ARG TAG=3.11

FROM ${BUILD_IMAGE}:${BUILD_TAG} as builder

COPY apt-install.sh requirements.txt docker-entrypoint.sh ./

ARG CMAKE_ARGS=""
ENV CMAKE_ARGS=${CMAKE_ARGS}
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM ${BASE_IMAGE}:${TAG}

COPY docker-entrypoint.sh ./
COPY --from=builder ./.local ./.local

ENTRYPOINT ["./docker-entrypoint.sh"]
EXPOSE 8000
