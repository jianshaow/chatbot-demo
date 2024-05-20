# llama-index

Demo how to use LlamaIndex.

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
# run with fastchat's OpenAI-Compatible API
export CHROMA_DB_DIR=chroma.local
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://host.docker.internal:8000/v1
~~~

## Docker Environment

### Build
~~~ shell
export image_ver=0.0.10
docker build -t jianshao/llamaindex-demo:$image_ver-cpu .
docker build -t jianshao/llamaindex-demo:$image_ver-gpu . \
       --build-arg TAG=2.2.1-gpu --build-arg REQUIREMENTS=requirements-gpu.txt
docker push jianshao/llamaindex-demo:$image_ver-cpu
docker push jianshao/llamaindex-demo:$image_ver-gpu
~~~
### Test
~~~ shell
docker run --name llamaindex-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache \
           -v $PWD:/workspaces/llama-index \
           jianshao/llamaindex-demo:$image_ver-gpu bash
~~~
