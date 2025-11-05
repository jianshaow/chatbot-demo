from common import hf_mm_model as model_name
from common.hgface import get_llm
from common.images import show_demo_image
from common.models import default_model_kwargs, demo_multi_modal

model_kwargs = default_model_kwargs()

show_demo_image()
mm_model = get_llm(model_name)
demo_multi_modal(mm_model, model_name, streaming=False)
