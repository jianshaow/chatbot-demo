from common import ollama_chat_model as model
from common.models import demo_chat
from common.ollama import get_llm

chat_model = get_llm(model)
demo_chat(chat_model, model)
