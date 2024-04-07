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
docker build -t jianshao/llama-cpp-server:cpu .
docker push jianshao/llama-cpp-server:cpu
docker build --build-arg BASE_IMAGE=jianshao/cuda-rt-base --build-arg TAG=12.3 \
             --build-arg LC_INDEX_ARG=https://abetlen.github.io/llama-cpp-python/whl/cu123 \
             -t jianshaollama-cpp-server:gpu .
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
