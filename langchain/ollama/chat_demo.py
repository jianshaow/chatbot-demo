from common import ollama_chat_model as model
from common.models import demo_chat
from common.ollama import get_chat_model

chat_model = get_chat_model(model)
demo_chat(chat_model, model)
