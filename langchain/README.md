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
export image_ver=0.0.3
docker build -t jianshao/langchain-demo:$image_ver .
docker push jianshao/langchain-demo:$image_ver
~~~
### Test
~~~ shell
docker run --name langchain-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache -v $PWD:/workspaces/langchain \
           jianshao/langchain-demo:$image_ver bash
~~~