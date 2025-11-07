from common import google_mm_model as model_name
from common.google import get_chat_model
from common.images import show_demo_image
from common.models import demo_multi_modal

show_demo_image()
mm_model = get_chat_model(model_name)
demo_multi_modal(mm_model, model_name)
