from common import google_embed_model as model_name
from common.google import get_embed_model
from common.models import demo_embed

embed_model = get_embed_model(model=model_name, transport="rest")
demo_embed(embed_model, model_name)
