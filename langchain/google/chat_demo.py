from common import google_chat_model as model_name
from common.google import get_chat_model
from common.models import demo_chat

chat_model = get_chat_model(model_name)
demo_chat(chat_model, model_name)
