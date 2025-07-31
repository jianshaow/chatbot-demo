from common import hf_chat_model as model_name
from common.hgface import get_llm
from common.models import demo_chat

chat_model = get_llm(model_name)
demo_chat(chat_model, model_name)
