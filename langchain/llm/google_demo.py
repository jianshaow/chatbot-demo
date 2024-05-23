import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a pirate with a colorful personality."), ("user", "{input}")]
)

model = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-latest")
llm = ChatGoogleGenerativeAI(model=model)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

print("-" * 80)
response = chain.invoke({"input": "What is your name?"})
for chunk in response:
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
