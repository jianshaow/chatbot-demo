import torch

from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="lmsys/vicuna-7b-v1.5",
    model_name="lmsys/vicuna-7b-v1.5",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    model_kwargs={"torch_dtype": torch.float16},
)

response = llm.complete("hello")
print(response.text)
