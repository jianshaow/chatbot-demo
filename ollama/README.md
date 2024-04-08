# ollama

## Docker Environment

### Build
~~~ shell
export ollama_ver=0.1.31
docker build -t jianshao/ollama-server:$ollama_ver-cpu .
docker push jianshao/ollama-server:$ollama_ver-cpu
~~~
### Test
~~~ shell
# run a ollama server
docker run --name ollama-server -it --rm -p 11434:11434 \
           -v $HOME/.ollama:/home/devel/.ollama \
           jianshao/ollama-server:$ollama_ver-cpu

# run a codellama with cli
docker exec -it ollama-server ollama run codellama

# run open-webui to verify
docker run -it --rm -p 3000:8080 --add-host=host.docker.internal:host-gateway \
           -v open-webui:/app/backend/data ghcr.io/open-webui/open-webui:main
~~~
