from openai import OpenAI

from common import openai_chat_model as model
from common.prompts import (
    chat_system_message as system_prompt,
    chat_question_message as question,
)

print("-" * 80)
print("chat model:", model)

client = OpenAI()
response = client.chat.completions.create(
    model=model, messages=[system_prompt, question], stream=True
)

print("-" * 80)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print("\n", "-" * 80, sep="")
