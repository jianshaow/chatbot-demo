from llama_index.llms.openai import OpenAI

from common import openai_chat_model as model
from common.models import demo_chat

chat_model = OpenAI(model=model)
demo_chat(chat_model)
