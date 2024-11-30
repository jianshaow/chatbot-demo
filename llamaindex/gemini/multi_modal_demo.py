from llama_index.multi_modal_llms.gemini import GeminiMultiModal

from common import google_mm_model as model_name
from common.images import show_demo_image
from common.models import demo_multi_modal

show_demo_image()
mm_model = GeminiMultiModal(model_name=model_name, transport="rest")
demo_multi_modal(mm_model, model_name)
