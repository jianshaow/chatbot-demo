from common import ollama_embed_model as embed_model_name
from common.models import demo_embed
from common.ollama import get_embed_model

embed_model = get_embed_model(model_name=embed_model_name)
demo_embed(embed_model, embed_model_name)
