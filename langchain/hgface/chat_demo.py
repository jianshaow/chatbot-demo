from common import hf_chat_model as model_name
from common.hgface import get_chat_model
from common.models import demo_chat

chat_model = get_chat_model(model_name)
demo_chat(chat_model, model_name)
