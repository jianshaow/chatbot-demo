# one-for-all

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

