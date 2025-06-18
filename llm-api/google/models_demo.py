import sys

from dotenv import load_dotenv

from google import genai

load_dotenv()

verbose = len(sys.argv) > 1 and sys.argv[1] == "verbose" or False

client = genai.Client()
models = client.models.list()
print("-" * 80)
for model in models:
    if verbose:
        print(model)
    else:
        print(model.name)
print("-" * 80)
