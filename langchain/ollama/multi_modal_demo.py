from langchain_ollama import ChatOllama

from common import ollama_base_url as base_url
from common import ollama_mm_model as model
from common.images import show_demo_image
from common.models import demo_multi_modal

_, image_data = show_demo_image()
mm_model = ChatOllama(base_url=base_url, model=model)
demo_multi_modal(mm_model, model, image_data)
