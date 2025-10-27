from typing import cast

from common import openai_mm_model as model
from common.openai import get_client
from common.prompts import mm_question_message
from openai import Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletionUserMessageParam

print("-" * 80)
print("multi-modal model:", model)

client = get_client()
user_message = cast(ChatCompletionUserMessageParam, mm_question_message)
response: Stream[ChatCompletionChunk] = client.chat.completions.create(
    model=model,
    messages=[user_message],
    stream=True,
)

print("-" * 80)
for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print("\n", "-" * 80, sep="")
