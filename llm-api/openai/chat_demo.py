from typing import cast

from common import openai_chat_model as model
from common.openai import get_client
from common.prompts import chat_question_message as question
from common.prompts import chat_system_message as system_prompt
from openai import Stream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

print("-" * 80)
print("chat model:", model)

client = get_client()
system_message = cast(ChatCompletionSystemMessageParam, system_prompt)
user_message = cast(ChatCompletionUserMessageParam, question)
response: Stream[ChatCompletionChunk] = client.chat.completions.create(
    model=model, messages=[system_message, user_message], stream=True
)

print("-" * 80)
for chunk in response:
    if len(chunk.choices) == 0:
        continue
    if content := chunk.choices[0].delta.content:
        print(content, end="")
print("\n", "-" * 80, sep="")
