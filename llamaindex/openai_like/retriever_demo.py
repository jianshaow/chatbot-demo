from common import openai_like_embed_model as model_name
from common.models import demo_retrieve
from common.openai_like import get_embed_model

embed_model = get_embed_model(model_name=model_name)
demo_retrieve(embed_model, model_name)
