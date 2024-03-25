# vllm

## Local Environment

### Prepare
~~~ shell
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
~~~

### Run
~~~ shell
python -m vllm.entrypoints.openai.api_server --model TheBloke/vicuna-13B-v1.5-AWQ
~~~

## Docker Environment

### Build
~~~ shell
docker build -t jianshao/vllm-base:gpu . -f Dockerfile.base
docker push jianshao/vllm-base:gpu

docker build -t jianshao/vllm-server:gpu . -f Dockerfile.server
docker push jianshao/vllm-server:gpu
~~~
### Test
~~~ shell
# run a openai api compatible server
docker run -it --rm -p 8000:8000 \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/vllm-server:cpu

# run a openai api compatible server on GPU
docker run -it --rm --gpus all -p 8000:8000 \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/vllm-server:gpu

# run a next chat to verify
docker run -it --rm -p 3000:3000 \
           -e BASE_URL=http://<docker-host>:8000 \
           yidadaa/chatgpt-next-web:v2.11.3
~~~
