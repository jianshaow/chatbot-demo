from langchain_google_genai import ChatGoogleGenerativeAI

from common import google_chat_model as model_name
from common.models import demo_chat

chat_model = ChatGoogleGenerativeAI(model=model_name, transport="rest")
demo_chat(chat_model, model_name)
