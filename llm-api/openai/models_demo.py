import sys

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

verbose = len(sys.argv) > 1 and sys.argv[1] == "verbose" or False

client = OpenAI()

models = client.models.list()
print("-" * 80)
for model in models:
    if model.owned_by != "openai":
        continue
    if verbose:
        print(model)
    else:
        print(model.id)
print("-" * 80)
