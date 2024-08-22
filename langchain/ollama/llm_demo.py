from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import ollama_base_url as base_url, ollama_chat_model as model

llm = ChatOllama(base_url=base_url, model=model)
print("-" * 80)
print("chat model:", model)

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a pirate with a colorful personality."), ("user", "{input}")]
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

print("-" * 80)
response = chain.stream({"input": "What is your name?"})
for chunk in response:
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
