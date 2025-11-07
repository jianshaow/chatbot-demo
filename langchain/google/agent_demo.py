from common import google_chat_model as chat_model_name
from common import google_embed_model as embed_model_name
from common.google import get_chat_model, get_embed_model
from common.models import demo_agent

embed_model = get_embed_model(embed_model_name)
chat_model = get_chat_model(chat_model_name)
demo_agent(embed_model, embed_model_name, chat_model, chat_model_name)
