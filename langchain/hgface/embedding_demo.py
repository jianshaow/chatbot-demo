from langchain_huggingface import HuggingFaceEmbeddings

from common import hf_embed_model as model_name
from common.models import demo_embed

embed_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)
demo_embed(embed_model, model_name)
