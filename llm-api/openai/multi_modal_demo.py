from common import openai_mm_model as model
from common.prompts import mm_question_message
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

print("-" * 80)
print("multi-modal model:", model)

client = OpenAI()
response: Stream[ChatCompletionChunk] = client.chat.completions.create(
    model=model,
    messages=[mm_question_message],
    stream=True,
)

print("-" * 80)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print("\n", "-" * 80, sep="")
