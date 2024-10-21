from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

models = client.models.list()
print("-" * 80)
for model in models:
    print(model)
print("-" * 80)
