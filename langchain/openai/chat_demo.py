from common import openai_chat_model as model_name
from common.models import demo_chat
from common.openai import get_llm

chat_model = get_llm(model=model_name)
demo_chat(chat_model, model_name)
