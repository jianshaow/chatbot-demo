from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a pirate with a colorful personality."), ("user", "{input}")]
)

llm = ChatGoogleGenerativeAI(model="gemini-pro")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

print("-" * 80)
response = chain.invoke({"input": "What is your name?"})
print(response)
print("-" * 80)
