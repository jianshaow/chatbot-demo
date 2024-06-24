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

## Docker Environment

### Build
~~~ shell
export llamaindex_ver=0.10.43
export torch_ver=2.3.1
docker build -t jianshao/llamaindex-demo:$llamaindex_ver-cpu . \
       --build-arg TAG=$torch_ver-cpu
docker build -t jianshao/llamaindex-demo:$llamaindex_ver-gpu . \
       --build-arg TAG=$torch_ver-gpu --build-arg REQUIREMENTS=requirements-gpu.txt
docker push jianshao/llamaindex-demo:$llamaindex_ver-cpu
docker push jianshao/llamaindex-demo:$llamaindex_ver-gpu
~~~
### Test
~~~ shell
docker run --name llamaindex-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache \
           -v $PWD:/workspaces/llama-index \
           jianshao/llamaindex-demo:$llamaindex_ver-gpu bash
~~~
