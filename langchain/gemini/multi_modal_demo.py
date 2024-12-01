from langchain_google_genai import ChatGoogleGenerativeAI

from common import google_mm_model as model_name
from common.images import show_demo_image
from common.models import demo_multi_modal

show_demo_image()
mm_model = ChatGoogleGenerativeAI(model=model_name, transport="rest")
demo_multi_modal(mm_model, model_name)
