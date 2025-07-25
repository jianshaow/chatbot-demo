from llama_index.llms.openai import OpenAI

from common import openai_mm_model as model
from common.images import show_demo_image
from common.models import demo_multi_modal

show_demo_image()
mm_model = OpenAI(model=model)
demo_multi_modal(mm_model, model)
