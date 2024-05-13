import os
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
model = os.environ.get("OLLAMA_MODEL", "vicuna:13b")
llm = Ollama(base_url=base_url, model=model)

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a pirate with a colorful personality."), ("user", "{input}")]
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

print("-" * 80)
response = chain.invoke({"input": "What is your name?"})
print(response)
print("-" * 80)
