# fastchat

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
export fschat_ver=0.2.35
docker build -t jianshao/fastchat-demo:$fschat_ver .
docker push jianshao/fastchat-demo:$fschat_ver
~~~
### Test image
~~~ shell
export model_path=TheBloke/vicuna-13B-v1.5-AWQ
# mount local model path
docker run -it --rm --gpus all \
           -v $HOME/huggingface/$model_path:/workspace/model \
           jianshao/fastchat-demo:$fschat_ver \
           python -m fastchat.serve.cli --model-path /workspace/model

# mount huggingface cache path
docker run -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/fastchat-demo:$fschat_ver \
           python -m fastchat.serve.cli --model-path $model_path
~~~

### Run as docker compose
~~~ shell
# for shared service
docker network create -d bridge fastchat-shared-network
~~~

## Minikube Environment

### setup minikube
~~~ shell
# set a specific minikube profile
export MINIKUBE_PROFILE=fastchat
# verify profile
minikube profile
# start an unlimited resources k8s for fastchat(for first start)
minikube start --driver docker --container-runtime docker --gpus all --cpus no-limit --memory no-limit
# mount the model path for reuse
minikube mount --uid=$(id -u) --gid=$(id -g) ~/models:/home/devel/.cache
# check the model path via ssh
minikube ssh 'ls -al /home/devel/.cache'
~~~

### prepare sources
~~~ shell
export MINIKUBE_PROFILE=fastchat
# prepare storage, env and service
kubectl apply -f storage.yaml
~~~
