from common import ollama_embed_model as model_name
from common.models import demo_retrieve
from common.ollama import get_embed_model

embed_model = get_embed_model(model_name)
demo_retrieve(embed_model, model_name)
