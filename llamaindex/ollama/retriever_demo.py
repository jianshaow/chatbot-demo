from common.ollama import NormOllamaEmbedding
from common import ollama_base_url as base_url, ollama_embed_model as model_name
from common.models import demo_recieve

embed_model = NormOllamaEmbedding(base_url=base_url, model_name=model_name)
demo_recieve(embed_model, model_name)
