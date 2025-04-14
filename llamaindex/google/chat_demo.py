from llama_index.llms.google_genai import GoogleGenAI

from common import gemini_chat_model as model
from common.models import demo_chat

chat_model = GoogleGenAI(model=model, transport="rest")
demo_chat(chat_model, model)
