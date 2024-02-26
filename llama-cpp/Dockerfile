ARG BASE_IMAGE=python
ARG TAG=3.11-slim

FROM ${BASE_IMAGE}:${TAG}

COPY apt-install.sh packages.txt requirements.txt docker-entrypoint.sh /
RUN /apt-install.sh /packages.txt && \
    useradd -ms /bin/bash devel
USER devel
ENV HOME=/home/devel
ENV PATH=$HOME/.local/bin:$PATH
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

ENTRYPOINT ["/docker-entrypoint.sh"]
EXPOSE 8000
