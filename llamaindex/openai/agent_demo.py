from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from common import openai_chat_model as chat_model_name
from common import openai_embed_model as embed_model_name
from common.models import demo_agent

embed_model = OpenAIEmbedding(model=embed_model_name)
chat_model = OpenAI(model=chat_model_name)
demo_agent(embed_model, chat_model)
