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
./build-image.sh
~~~
### Test
~~~ shell
docker run --name llamaindex-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache \
           -v $PWD:/workspaces/llama-index \
           jianshao/llamaindex-demo:latest bash
~~~
