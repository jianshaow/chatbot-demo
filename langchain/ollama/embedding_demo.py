from langchain_ollama import OllamaEmbeddings

from common import ollama_base_url as base_url, ollama_embed_model as model_name
from common.models import demo_embed

embed_model = OllamaEmbeddings(base_url=base_url, model=model_name)
demo_embed(embed_model, model_name)
