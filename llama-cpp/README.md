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
export image_ver=0.0.1
docker build -t jianshao/llama-cpp-demo:${image_ver}-cpu .
docker push jianshao/llama-cpp-demo:${image_ver}-cpu
~~~
### Test
~~~ shell
# run a openai api compatible server
docker run --name llama-cpp-demo -it --rm --gpus all \
           -p 8000:8000 \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/llama-cpp-demo:${image_ver}-cpu

# run a next chat to verify
docker run --name nextchat -it --rm \
           -p 3000:3000 \
           -e BASE_URL=http://<docker-host>:8000 \
           yidadaa/chatgpt-next-web:v2.10.1
~~~
