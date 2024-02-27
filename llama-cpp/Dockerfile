FROM jianshao/dl-base:3.11

COPY apt-install.sh packages.txt requirements.txt docker-entrypoint.sh ./
USER root
RUN ./apt-install.sh packages.txt
USER devel

ARG PIP_INDEX_URL=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["./docker-entrypoint.sh"]
EXPOSE 8000
