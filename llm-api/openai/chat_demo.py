from common import openai_chat_model as model
from common.prompts import chat_question_message as question
from common.prompts import chat_system_message as system_prompt
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

print("-" * 80)
print("chat model:", model)

client = OpenAI()
response: Stream[ChatCompletionChunk] = client.chat.completions.create(
    model=model, messages=[system_prompt, question], stream=True
)

print("-" * 80)
for chunk in response:
    if content := chunk.choices[0].delta.content:
        print(content, end="")
print("\n", "-" * 80, sep="")
