import requests, os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

image_url = "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"

img_response = requests.get(image_url)
print(image_url)
img = Image.open(BytesIO(img_response.content))
plt.imshow(img)
plt.show()

message = HumanMessage(
    content=[
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
        {
            "type": "text",
            "text": "{input}",
        },
    ]
)

prompt = ChatPromptTemplate.from_messages([message])

model = os.environ.get("GEMINI_MODEL", "models/gemini-pro-vision")
llm = ChatGoogleGenerativeAI(model=model)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

print("-" * 80)

input = "Identify the city where this photo was taken."
print("Question:", input)
print("Answer:", end="")
response = chain.invoke({"input": input})
print("Answer:", response)

print("-" * 80)

input = "Give me more context for this image."
print("Question:", input)
print("Answer:", end="")
response = chain.invoke({"input": input})
print("Answer:", response)

print("-" * 80)
