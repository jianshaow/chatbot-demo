FROM python:3.11-slim

RUN useradd -ms /bin/bash devel
USER devel
ENV HOME=/home/devel
ENV PATH=$HOME/.local/bin:$PATH
WORKDIR $HOME

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
