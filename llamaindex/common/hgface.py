from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

from common.models import default_model_kwargs


def get_llm(model_name: str):
    model_kwargs = default_model_kwargs()

    llm = HuggingFaceLLM(
        tokenizer_name=model_name, model_name=model_name, model_kwargs=model_kwargs
    )
    llm.generate_kwargs["pad_token_id"] = llm._tokenizer.eos_token_id
    return llm


def get_embed_model(model_name: str):
    return HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)
