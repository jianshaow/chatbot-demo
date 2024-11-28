from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import hf_chat_model as model_name
from common.models import default_model_kwargs

model_kwargs = default_model_kwargs()

llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs=model_kwargs,
    pipeline_kwargs={"max_new_tokens": 512},
)
chat_model = ChatHuggingFace(llm=llm)
print("-" * 80)
print("chat model:", model_name)

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a pirate with a colorful personality."), ("user", "{input}")]
)
output_parser = StrOutputParser()
chain = prompt | chat_model | output_parser

print("-" * 80)
response = chain.stream({"input": "What is your name?"})
for chunk in response:
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
