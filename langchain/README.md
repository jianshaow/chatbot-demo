# langchain

Demo how to use LangChain.

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
export torch_ver=2.3.1
./build-image.sh
~~~
### Test
~~~ shell
docker run --name langchain-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache -v $PWD:/workspaces/langchain \
           jianshao/langchain-demo:latest bash
~~~