from llama_index.embeddings.openai import OpenAIEmbedding

from common import openai_embed_model as model_name
from common.models import demo_retrieve

embed_model = OpenAIEmbedding(model=model_name)
demo_retrieve(embed_model)
