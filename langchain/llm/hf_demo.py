import os
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a pirate with a colorful personality."), ("user", "{input}")]
)

model_name = os.environ.get("HF_MODEL", "lmsys/vicuna-7b-v1.5")
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

print("-" * 80)
response = chain.stream({"input": "What is your name?"})
for chunk in response:
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
