from llama_index.llms.ollama import Ollama

from common import ollama_base_url as base_url, ollama_chat_model as model
from common.models import demo_chat

chat_model = Ollama(base_url=base_url, model=model)
demo_chat(chat_model, model)
