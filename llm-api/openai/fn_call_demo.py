import json

from clients import get_client

from common import openai_fc_model as model
from common.fn_tools import tools
from common.functions import fns
from common.prompts import fn_call_adv_question_message as question

print("-" * 80)
print("fn call model:", model)

client = get_client()

messages = [question]
response = openai.chat.completions.create(model=model, messages=messages, tools=tools) # type: ignore

while response.choices[0].finish_reason == "tool_calls":
    print("-" * 80)
    messages.append(response.choices[0].message) # type: ignore
    for tool_call in response.choices[0].message.tool_calls: # type: ignore
        fn = fns[tool_call.function.name]
        fn_args = json.loads(tool_call.function.arguments)
        print("=== Calling Function ===")
        print(
            "Calling function:",
            tool_call.function.name,
            "with args:",
            tool_call.function.arguments,
        )
        fn_result = fn(**fn_args)
        print("Got output:", fn_result)
        print("========================\n")
        messages.append(
            {
                "role": "tool",
                "content": str(fn_result),
                "tool_call_id": tool_call.id,
            }
        )
<<<<<<< HEAD
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools  # type: ignore
=======
    response = openai.chat.completions.create(
        model=model, messages=messages, tools=tools # type: ignore
>>>>>>> fcba468 (add type ignore)
    )

print("-" * 80)
print(response.choices[0].message.content)
