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
export langchain_ver=0.2.2
docker build -t jianshao/langchain-demo:$langchain_ver-cpu . \
       --build-arg TAG=2.3.0-cpu
docker build -t jianshao/langchain-demo:$langchain_ver-gpu . \
       --build-arg TAG=2.3.0-gpu --build-arg REQUIREMENTS=requirements-gpu.txt
docker push jianshao/langchain-demo:$langchain_ver-cpu
docker push jianshao/langchain-demo:$langchain_ver-gpu
~~~
### Test
~~~ shell
docker run --name langchain-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache -v $PWD:/workspaces/langchain \
           jianshao/langchain-demo:$langchain_ver bash
~~~