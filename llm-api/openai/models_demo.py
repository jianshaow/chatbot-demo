from openai import OpenAI

client = OpenAI()

models = client.models.list()
print("-" * 80)
for model in models:
    print(model)
print("-" * 80)
