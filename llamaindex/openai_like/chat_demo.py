from common import openai_like_chat_model as model
from common.models import demo_chat
from common.openai_like import get_llm

chat_model = get_llm(model=model)
demo_chat(chat_model, model)
