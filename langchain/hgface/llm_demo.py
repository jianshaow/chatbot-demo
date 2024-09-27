import os, torch
from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import hf_chat_model as model_name

prompt = PromptTemplate(
    template="You are a pirate with a colorful personality. USER: {input}",
    input_variables=["input"],
)

model_kwargs = {}
if os.getenv("BNB_ENABLED", "false") == "true":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs["quantization_config"] = quantization_config

llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs=model_kwargs,
)
print("-" * 80)
print("chat model:", model_name)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

print("-" * 80)
response = chain.stream({"input": "What is your name?"})
for chunk in response:
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
