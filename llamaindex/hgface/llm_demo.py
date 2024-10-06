from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage

from common import hf_chat_model as model_name
from common.models import default_model_kwargs

model_kwargs = default_model_kwargs()

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    tokenizer_name=model_name,
    model_name=model_name,
    model_kwargs=model_kwargs,
)
print("-" * 80)
print("chat model:", model_name)

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality."),
    ChatMessage(role="user", content="What is your name?"),
]
print("-" * 80)
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
print("\n", "-" * 80, sep="")
