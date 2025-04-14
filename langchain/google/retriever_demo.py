from langchain_google_genai import GoogleGenerativeAIEmbeddings

from common import google_embed_model as model_name
from common.models import demo_retrieve

embed_model = GoogleGenerativeAIEmbeddings(model=model_name, transport="rest")
demo_retrieve(embed_model, model_name)
