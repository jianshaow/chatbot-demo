from common import openai_like_chat_model as chat_model_name
from common import openai_like_embed_model as embed_model_name
from common.models import demo_agent
from common.openai_like import get_embed_model, get_llm

embed_model = get_embed_model(model_name=embed_model_name)
chat_model = get_llm(model=chat_model_name)
demo_agent(embed_model, chat_model)
