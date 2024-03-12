# LlamaIndex

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
export CHROMA_DB_DIR=LlamaIndex/local.chroma_db
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://localhost:8000/v1
~~~

## Docker Environment

### Build
~~~ shell
export image_ver=0.0.6
docker build -t jianshao/llamaindex-demo:$image_ver .
docker push jianshao/llamaindex-demo:$image_ver
~~~
### Test
~~~ shell
docker run --name llamaindex-demo -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache \
           -v $PWD:/workspaces/LlamaIndex \
           jianshao/llamaindex-demo:$image_ver bash
~~~
