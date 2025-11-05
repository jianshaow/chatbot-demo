from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)

from common.models import default_model_kwargs
from common.transformers import trfs_pipeline


def get_chat_model(model_name: str, **kwargs) -> ChatHuggingFace:
    model_kwargs = default_model_kwargs()

    llm = HuggingFacePipeline(
        pipeline=trfs_pipeline(model_name, model_kwargs), **kwargs
    )
    return ChatHuggingFace(llm=llm)


def get_embed_model(model_name: str, **kwargs) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
        **kwargs
    )
