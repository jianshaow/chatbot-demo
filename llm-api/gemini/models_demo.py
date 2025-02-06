import sys

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types import ModelsIterable

load_dotenv()

genai.configure(transport="rest")

verbose = len(sys.argv) > 1 and sys.argv[1] == "verbose" or False

models: ModelsIterable = genai.list_models()
print("-" * 80)
for model in models:
    if verbose:
        print(model)
    else:
        print(model.name)
print("_" * 80)
