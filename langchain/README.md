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
export langchain_core_ver=0.2.27
export torch_ver=2.3.1
docker build -t jianshao/langchain-demo:$langchain_core_ver-cpu . \
       --build-arg TAG=$torch_ver-cpu
docker build -t jianshao/langchain-demo:$langchain_core_ver-gpu . \
       --build-arg TAG=$torch_ver-gpu --build-arg REQUIREMENTS=requirements-gpu.txt
docker push jianshao/langchain-demo:$langchain_core_ver-cpu
docker push jianshao/langchain-demo:$langchain_core_ver-gpu
~~~
### Test
~~~ shell
docker run --name langchain-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache -v $PWD:/workspaces/langchain \
           jianshao/langchain-demo:$langchain_ver bash
~~~