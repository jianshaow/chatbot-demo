from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import google_chat_model as model

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a pirate with a colorful personality."), ("user", "{input}")]
)

chat_model = ChatGoogleGenerativeAI(model=model, transport="rest")
print("-" * 80)
print("chat model:", model)

output_parser = StrOutputParser()
chain = prompt | chat_model | output_parser

print("-" * 80)
response = chain.stream({"input": "What is your name?"})
for chunk in response:
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
