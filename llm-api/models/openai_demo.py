from openai import OpenAI

client = OpenAI()

models = client.models.list()
for model in models:
    print(model)
