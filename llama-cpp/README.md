# llama-cpp

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
python -m llama_cpp.server --host 0.0.0.0 --chat_format chatml
~~~

## Docker Environment

### Build
~~~ shell
export cpu_image_ver=cpu-0.0.1
docker build -t jianshao/llama-cpp-demo:$cpu_image_ver .
docker push jianshao/llama-cpp-demo:$cpu_image_ver
~~~
### Test
~~~ shell
# run a openai api compatible server
docker run --name llama-cpp-demo -it --rm --gpus all \
           -p 8000:8000 \
           -v $HOME/models:/home/devel/models \
           -v $PWD:/workspaces/llama-cpp \
           ghcr.io/abetlen/llama-cpp-python:v0.2.51 \
           python -m llama_cpp.server --host 0.0.0.0 --chat_format chatml \
           --model /home/devel/models/vicuna-13b-v1.5.Q4_K_M-HF/vicuna-13b-v1.5.Q4_K_M.gguf

# run a next chat to verify
docker run --name nextchat -it --rm all \
           -p 3000:3000 \
           -e BASE_URL=http://<docker-host>:8000 \
           yidadaa/chatgpt-next-web:v2.10.1
~~~
