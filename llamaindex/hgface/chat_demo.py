from llama_index.llms.huggingface import HuggingFaceLLM

from common import hf_chat_model as model_name
from common.models import default_model_kwargs, demo_chat

model_kwargs = default_model_kwargs()

chat_model = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    tokenizer_name=model_name,
    model_name=model_name,
    model_kwargs=model_kwargs,
)
demo_chat(chat_model, model_name)
