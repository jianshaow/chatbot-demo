from common import google_embed_model as model_name
from common.google import get_embed_model
from common.models import demo_retrieve

embed_model = get_embed_model(model_name)
demo_retrieve(embed_model, model_name)
