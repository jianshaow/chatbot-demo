import sys

from ollama import Client

client = Client()

verbose = len(sys.argv) > 1 and sys.argv[1] == "verbose" or False

models = client.list()["models"]
print("-" * 80)
for model in models:
    if verbose:
        print(model)
    else:
        print(model["model"])
print("_" * 80)
