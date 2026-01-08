from llama_index.llms.google_genai import GoogleGenAI

from common import google_mm_model as model
from common.images import show_demo_image
from common.models import demo_multi_modal

show_demo_image()
mm_model = GoogleGenAI(model=model)
demo_multi_modal(mm_model)
