from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import hf_chat_model as model_name
from common.models import default_model_kwargs

prompt = PromptTemplate(
    template="You are a pirate with a colorful personality. USER: {input}",
    input_variables=["input"],
)

model_kwargs = default_model_kwargs()

llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs=model_kwargs,
    pipeline_kwargs={"max_new_tokens": 256},
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
