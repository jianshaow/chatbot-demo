from llama_index.llms.gemini import Gemini

from common import gemini_chat_model as model
from common.models import demo_chat

chat_model = Gemini(model=model, transport="rest")
demo_chat(chat_model, model)
