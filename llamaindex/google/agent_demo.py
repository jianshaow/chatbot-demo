from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from common import google_embed_model as embed_model_name
from common import google_chat_model as chat_model_name
from common.models import demo_agent

embed_model = GoogleGenAIEmbedding(model_name=embed_model_name)
chat_model = GoogleGenAI(model=chat_model_name)
demo_agent(embed_model, embed_model_name, chat_model, chat_model_name)
