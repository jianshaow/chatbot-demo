from common import hf_embed_model as model_name
from common.hgface import get_embed_model
from common.models import demo_retrieve

embed_model = get_embed_model(model_name)
demo_retrieve(embed_model, model_name)
