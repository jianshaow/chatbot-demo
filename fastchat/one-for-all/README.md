# one-for-all

## local environment
~~~ shell
python -m fastchat.serve.controller
python -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
python -m fastchat.serve.gradio_web_server
~~~

## docker compose environment
~~~ shell
cp .env.template .env
docker-compose up -d
~~~
## kubernetes environment
~~~ shell
# for cpu only
kubectl apply -f deploy.cpu.yaml
# for gpu
kubectl apply -f deploy.gpu.yaml
~~~

