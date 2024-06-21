import os
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
model_name = os.environ.get("OLLAMA_MODEL", "vicuna:13b")
llm = ChatOllama(base_url=base_url, model=model_name)
print("-" * 80)
print("chat model:", model_name)

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
