# llama-cpp

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
python -m llama_cpp.server --host 0.0.0.0 --chat_format chatml
~~~

## Docker Environment

### Build
~~~ shell
docker build -t jianshao/llama-cpp-base:cpu . -f Dockerfile.base
docker push jianshao/llama-cpp-base:cpu
docker build --build-arg BUILD_IMAGE=jianshao/cuda-builder --build-arg BUILD_TAG=12.3 \
             --build-arg BASE_IMAGE=jianshao/cuda-rt-base --build-arg TAG=12.3 \
             --build-arg CMAKE_ARGS="-DLLAMA_CUBLAS=on" \
             -t jianshao/llama-cpp-base:gpu . -f Dockerfile.base
docker push jianshao/llama-cpp-base:gpu

docker build -t jianshao/llama-cpp-server:cpu . -f Dockerfile.server
docker push jianshao/llama-cpp-server:cpu
docker build --build-arg TAG=gpu \
             -t jianshao/llama-cpp-server:gpu . -f Dockerfile.server
docker push jianshao/llama-cpp-server:gpu
~~~
### Test
~~~ shell
# run a openai api compatible server
docker run -it --rm -p 8000:8000 \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/llama-cpp-server:cpu

# run a openai api compatible server on GPU
docker run -it --rm --gpus all -p 8000:8000 \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/llama-cpp-server:gpu

# run a next chat to verify
docker run -it --rm -p 3000:3000 \
           -e BASE_URL=http://<docker-host>:8000 \
           yidadaa/chatgpt-next-web:v2.11.3
~~~
