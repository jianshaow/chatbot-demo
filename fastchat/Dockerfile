FROM jianshao/torch-rt-base:2.2.1-gpu

ARG REQUIREMENTS=requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${REQUIREMENTS}
