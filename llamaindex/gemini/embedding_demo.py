from llama_index.embeddings.gemini import GeminiEmbedding

from common import gemini_embed_model as model_name
from common.models import demo_embed

embed_model = GeminiEmbedding(model_name=model_name, transport="rest")
demo_embed(embed_model, model_name)
