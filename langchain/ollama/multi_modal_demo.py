from langchain_ollama import ChatOllama

from common import ollama_base_url as base_url, ollama_mm_model as model_name
from common.images import show_demo_image
from common.models import demo_multi_modal

show_demo_image()
mm_model = ChatOllama(base_url=base_url, model=model_name)
demo_multi_modal(mm_model, model_name)
