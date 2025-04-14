from langchain_google_genai import GoogleGenerativeAIEmbeddings

from common import gemini_embed_model as model_name
from common.models import demo_embed

embed_model = GoogleGenerativeAIEmbeddings(model=model_name, transport="rest")
demo_embed(embed_model, model_name)
