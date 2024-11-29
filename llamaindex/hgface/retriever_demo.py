from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from common import hf_embed_model as model_name
from common.models import demo_recieve

embed_model = HuggingFaceEmbedding(model_name, trust_remote_code=True)
demo_recieve(embed_model, model_name)
