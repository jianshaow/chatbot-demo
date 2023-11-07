# fastchat

## build
~~~ shell
export image_ver=0.0.1
docker build -t jianshao/fastchat-demo:$image_ver .
docker push jianshao/fastchat-demo:$image_ver
~~~
## Test
~~~ shell
docker run --name fastchat-cli \
           -v $HOME/huggingface\LinkSoul/Chinese-Llama-2-7b-4bit:/workspace/model \
           jianshao/fastchat-demo:$image_ver \
           python -m fastchat.serve.cli --model-path /workspace/model
~~~
