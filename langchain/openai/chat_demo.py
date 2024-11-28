from langchain_openai import ChatOpenAI

from common import openai_chat_model as model_name
from common.models import demo_chat

chat_model = ChatOpenAI(model=model_name)
demo_chat(chat_model, model_name)
