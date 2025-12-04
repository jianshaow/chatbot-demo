from common import ollama_chat_model as model
from common import think
from common.prompts import chat_question_message as question
from common.prompts import chat_system_message as system_prompt
from ollama import Client

print("-" * 80)
print("chat model:", model)

client = Client()

response = client.chat(
    model=model, messages=[system_prompt, question], think=think, stream=True
)

print("-" * 80)
for chunk in response:
    print(chunk.message.content, end="", flush=True)
print("\n", "-" * 80, sep="")
