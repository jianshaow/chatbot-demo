from llama_index.llms.google_genai import GoogleGenAI

from common import gemini_mm_model as model
from common.images import show_demo_image
from common.models import demo_multi_modal

show_demo_image()
mm_model = GoogleGenAI(model=model, transport="rest")
demo_multi_modal(mm_model, model)
