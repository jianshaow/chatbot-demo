from langchain_openai import OpenAIEmbeddings

from common import openai_embed_model as model_name
from common.models import demo_retrieve

embed_model = OpenAIEmbeddings(model=model_name)
demo_retrieve(embed_model, model_name)
