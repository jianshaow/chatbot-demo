from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import openai_chat_model as model

chat_model = ChatOpenAI(model=model)
print("-" * 80)
print("chat model:", model)

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
