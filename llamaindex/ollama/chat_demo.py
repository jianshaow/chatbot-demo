from common import ollama_chat_model as chat_model_name
from common.models import demo_chat
from common.ollama import get_llm

chat_model = get_llm(chat_model_name)
demo_chat(chat_model)
