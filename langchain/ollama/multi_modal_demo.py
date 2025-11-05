from common import ollama_mm_model as model
from common.images import show_demo_image
from common.models import demo_multi_modal
from common.ollama import get_chat_model

_, image_data = show_demo_image()
mm_model = get_chat_model(model)
demo_multi_modal(mm_model, model, image_data)
