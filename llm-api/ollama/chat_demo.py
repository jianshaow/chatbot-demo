import ollama

from common import ollama_chat_model as model
from common.prompts import (
    chat_system_message as system_prompt,
    chat_question_message as question,
)

print("-" * 80)
print("chat model:", model)

response = ollama.chat(model=model, messages=[system_prompt, question], stream=True)

print("-" * 80)
for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)
print("\n", "-" * 80, sep="")
