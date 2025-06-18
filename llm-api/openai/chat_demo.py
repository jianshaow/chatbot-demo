from clients import get_client

from common import openai_chat_model as model
from common.prompts import chat_question_message as question
from common.prompts import chat_system_message as system_prompt
from openai import Stream
from openai.types.chat import ChatCompletionChunk

print("-" * 80)
print("chat model:", model)

client = get_client()
response: Stream[ChatCompletionChunk] = client.chat.completions.create(
    model=model, messages=[system_prompt, question], stream=True # type: ignore
) # type: ignore

print("-" * 80)
for chunk in response:
    if content := chunk.choices[0].delta.content:
        print(content, end="")
print("\n", "-" * 80, sep="")
