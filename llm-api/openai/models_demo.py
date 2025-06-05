import sys

from clients import get_client
from dotenv import load_dotenv

load_dotenv()

verbose = len(sys.argv) > 1 and sys.argv[1] == "verbose" or False

client = get_client()
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
