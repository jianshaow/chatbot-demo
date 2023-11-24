# fastchat

## Local Environment

### Prepare
~~~ shell
python -m venv --system-site-packages fastchat
source fastchat/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
~~~

## Docker Environment

### Build
~~~ shell
export image_ver=0.0.2
docker build -t jianshao/fastchat-demo:$image_ver .
docker push jianshao/fastchat-demo:$image_ver
~~~
### Test image
~~~ shell
export model_path=lmsys/vicuna-7b-v1.5
# mount local model path
docker run --name fastchat-cli -it --rm --gpus all \
           -v $HOME/huggingface/$model_path:/workspace/model \
           jianshao/fastchat-demo:$image_ver \
           python -m fastchat.serve.cli --model-path /workspace/model

# mount huggingface cache path
docker run --name fastchat-cli -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/fastchat-demo:$image_ver \
           python -m fastchat.serve.cli --model-path $model_path
~~~

### Run as docker compose
~~~ shell
// for shared service
docker network create -d bridge fastchat-shared-network
~~~
