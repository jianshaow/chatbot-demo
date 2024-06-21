import requests, os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

image_url = "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"

img_response = requests.get(image_url)
print(image_url)
img = Image.open(BytesIO(img_response.content))
plt.imshow(img)
plt.show()

template = HumanMessagePromptTemplate.from_template(
    [{"text": "{input}"}, {"image_url": {"url": "{image_url}"}}]
)
prompt = ChatPromptTemplate.from_messages([template])

model_name = os.environ.get("OPENAI_MM_MODEL", "gpt-4-vision-preview")
llm = ChatOpenAI(model=model_name)
print("-" * 80)
print("multi-modal model:", model_name)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

print("-" * 80)

input = "Identify the city where this photo was taken."
print("Question:", input)
response = chain.invoke({"input": input, "image_url": image_url})
print("Answer:", response)
print("-" * 80)

input = "Give me more context for this image."
print("Question:", input)
print("Answer:", end="")
response = chain.stream({"input": input, "image_url": image_url})
for chunk in response:
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
