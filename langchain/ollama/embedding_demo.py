from common import ollama_embed_model as model_name
from common.models import demo_embed
from common.ollama import get_embed_model

embed_model = get_embed_model(model_name)
demo_embed(embed_model, model_name)
