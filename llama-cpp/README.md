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
docker build -t jianshao/llama-cpp-demo:cpu .
docker push jianshao/llama-cpp-demo:cpu
docker build --build-arg BUILD_IMAGE=jianshao/cuda-builder --build-arg BUILD_TAG=12.3 \
             --build-arg BASE_IMAGE=jianshao/cuda-rt-base --build-arg TAG=12.3 \
             --build-arg CMAKE_ARGS="-DLLAMA_CUBLAS=on" -t jianshao/llama-cpp-demo:gpu .
docker push jianshao/llama-cpp-demo:gpu
~~~
### Test
~~~ shell
# run a openai api compatible server
docker run -it --rm -p 8000:8000 \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/llama-cpp-demo:${image_ver}-cpu

# run a openai api compatible server on GPU
docker run -it --rm --gpus all -p 8000:8000 \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/llama-cpp-demo:${image_ver}-gpu

# run a next chat to verify
docker run -it --rm -p 3000:3000 \
           -e BASE_URL=http://<docker-host>:8000 \
           yidadaa/chatgpt-next-web:v2.11.2
~~~
