from langchain_ollama import OllamaEmbeddings

from common import ollama_base_url as base_url
from common import ollama_chat_model as chat_model_name
from common import ollama_embed_model as embed_model_name
from common.models import demo_agent
from common.ollama import get_llm_model

embed_model = OllamaEmbeddings(base_url=base_url, model=embed_model_name)
chat_model = get_llm_model(model=chat_model_name)
demo_agent(embed_model, embed_model_name, chat_model, chat_model_name)
