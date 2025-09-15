from common import ollama_chat_model as chat_model_name
from common import ollama_embed_model as embed_model_name
from common.models import demo_agent
from common.ollama import get_embed_model, get_llm

embed_model = get_embed_model(embed_model_name)
chat_model = get_llm(chat_model_name)
demo_agent(embed_model, embed_model_name, chat_model, chat_model_name)
