from common import openai_fc_model as model
from common import openai_like_mm_model as model
from common.images import show_demo_image
from common.models import demo_multi_modal
from common.openai_like import get_llm

mm_model = get_llm(model=model)
show_demo_image()
demo_multi_modal(mm_model)
