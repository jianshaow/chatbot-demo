from langchain_ollama import OllamaEmbeddings

from common import ollama_base_url as base_url
from common.models import demo_retrieve

model_name = "paraphrase-multilingual:278m"
embed_model = OllamaEmbeddings(base_url=base_url, model=model_name)
demo_retrieve(embed_model, model_name, "data/zh-text", "地球发动机都安装在哪里？")
