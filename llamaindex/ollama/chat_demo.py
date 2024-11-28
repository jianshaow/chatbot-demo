from llama_index.llms.ollama import Ollama

from common import ollama_base_url as base_url, ollama_chat_model as model_name
from common.models import demo_chat

chat_model = Ollama(base_url=base_url, model=model_name)
demo_chat(chat_model, model_name)
