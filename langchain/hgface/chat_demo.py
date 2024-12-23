from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from common import hf_chat_model as model_name, add_method_kwargs
from common.models import default_model_kwargs, demo_chat
from common.transformers import trfs_pipeline

model_kwargs = default_model_kwargs()

llm = HuggingFacePipeline(pipeline=trfs_pipeline(model_name, model_kwargs))
chat_model = ChatHuggingFace(llm=llm)
add_method_kwargs(chat_model, "_generate", skip_prompt=True)
demo_chat(chat_model, model_name)
