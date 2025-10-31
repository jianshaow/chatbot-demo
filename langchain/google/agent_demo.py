from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from common import google_chat_model as chat_model_name
from common import google_embed_model as embed_model_name
from common.models import demo_agent

embed_model = GoogleGenerativeAIEmbeddings(model=embed_model_name, transport="rest")
chat_model = ChatGoogleGenerativeAI(model=chat_model_name, transport="rest")
demo_agent(embed_model, embed_model_name, chat_model, chat_model_name)
