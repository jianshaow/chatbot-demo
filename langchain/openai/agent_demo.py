from common import openai_chat_model as chat_model_name
from common import openai_embed_model as embed_model_name
from common.models import demo_agent
from common.openai import get_embed_model, get_chat_model

embed_model = get_embed_model(model=embed_model_name)
chat_model = get_chat_model(model=chat_model_name)
demo_agent(embed_model, embed_model_name, chat_model, chat_model_name)
