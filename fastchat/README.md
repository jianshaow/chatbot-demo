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
# for shared service
docker network create -d bridge fastchat-shared-network
~~~

## Minikube Environment

### setup minikube
~~~ shell
# start an unlimited resources k8s for fastchat
minikube -p fastchat start --driver docker --container-runtime docker --gpus all --cpus no-limit --memory no-limit
# mount the model path for reuse
minikube -p fastchat mount --uid=$(id -u) --gid=$(id -g) ~/models:/home/devel/.cache
# check the model path via ssh
minikube -p fastchat ssh 'ls -al /home/devel'
~~~

### prepare sources
~~~ shell
# prepare storage, env and service
kubectl apply -f storage.yaml
kubectl apply -f env-cm.yaml
kubectl apply -f service.yaml
# expose the service
minikube -p fastchat service -ndemo fastchat-demo
~~~
