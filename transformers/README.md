# transformers

Demo how to use HuggingFace transformers.

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
export transformers_ver=4.41.2
export torch_ver=2.3.1
docker build -t jianshao/transformers-demo:$transformers_ver-cpu . \
       --build-arg TAG=$torch_ver-cpu
docker build -t jianshao/transformers-demo:$transformers_ver-gpu . \
       --build-arg TAG=$torch_ver-gpu --build-arg REQUIREMENTS=requirements-gpu.txt
docker push jianshao/transformers-demo:$transformers_ver-cpu
docker push jianshao/transformers-demo:$transformers_ver-gpu
~~~
### Test
~~~ shell
docker run --name transformers-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache -v $PWD:/workspaces/transformers \
           jianshao/transformers-demo:$transformers_ver-gpu bash
~~~