from llama_index.llms.gemini import Gemini

from common import google_chat_model as model_name
from common.models import demo_chat

chat_model = Gemini(model_name=model_name, transport="rest")
demo_chat(chat_model, model_name)
