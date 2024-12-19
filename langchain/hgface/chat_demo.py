from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from common import hf_chat_model as model_name, add_kwargs
from common.models import default_model_kwargs, demo_chat

model_kwargs = default_model_kwargs()

llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs=model_kwargs,
    pipeline_kwargs={"max_new_tokens": 512},
)
chat_model = ChatHuggingFace(llm=llm)
chat_model._generate = add_kwargs(
    chat_model._generate,
    skip_prompt=True,
)
demo_chat(chat_model, model_name)
