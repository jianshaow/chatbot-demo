import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate

SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

model_name = "lmsys/vicuna-7b-v1.5"

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=model_name,
    model_name=model_name,
    device_map="auto",
    # is_chat_model=True,
    model_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
)

response = llm.complete("who are you")
print(response.text)
