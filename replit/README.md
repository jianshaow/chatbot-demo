# replit

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
export image_tag=0.0.1
docker build -t jianshao/replit-demo:$image_tag .
docker push jianshao/replit-demo:$image_tag
~~~
### Test image
~~~ shell
# mount huggingface cache path
docker run -it --rm --gpus all \
           -v $HOME/.cache:/home/devel/.cache \
           jianshao/replit-demo:$image_tag bash
~~~
