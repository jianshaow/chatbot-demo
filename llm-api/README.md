# llm-api

Demo how to use llm api.

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
export images_ver=0.0.1
docker build -t jianshao/llm-api-demo:$images_ver .
docker push jianshao/llm-api-demo:$images_ver
~~~
### Test
~~~ shell
docker run --name llm-api-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache -v $PWD:/workspaces/llm-api \
           jianshao/llm-api-demo:$images_ver bash
~~~