from common import openai_mm_model as model_name
from common.images import show_demo_image
from common.models import demo_multi_modal
from common.openai import get_llm

show_demo_image()
mm_model = get_llm(model=model_name)
demo_multi_modal(mm_model, model_name)
