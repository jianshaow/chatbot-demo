from langchain_huggingface import HuggingFaceEmbeddings

from common.models import demo_retrieve

model_name = "BAAI/bge-small-zh-v1.5"
embed_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)
demo_retrieve(embed_model, model_name, "data/zh-text", "地球发动机都安装在哪里？")
