import sys

import ollama

verbose = len(sys.argv) > 1 and sys.argv[1] == "verbose" or False

models = ollama.list()["models"]
print("-" * 80)
for model in models:
    if verbose:
        print(model)
    else:
        print(model["model"])
print("_" * 80)
