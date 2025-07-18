import os
import sys

from dotenv import load_dotenv

from common.openai import get_client

load_dotenv()

verbose = len(sys.argv) > 1 and sys.argv[1] == "verbose" or False
owned_by = os.getenv("OWNED_BY", "openai")

client = get_client()
models = client.models.list()
print("-" * 80)
for model in models:
    if model.owned_by != owned_by:
        continue
    if verbose:
        print(model)
    else:
        print(model.id)
print("-" * 80)
