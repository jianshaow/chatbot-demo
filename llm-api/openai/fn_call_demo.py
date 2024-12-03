import json, openai

from common.functions import fns
from common.fn_tools import tools

from common import openai_fc_model as model
from common.prompts import fn_call_question_message as question

print("-" * 80)
print("fn call model:", model)

messages = [question]
response = openai.chat.completions.create(model=model, messages=messages, tools=tools)

while response.choices[0].finish_reason == "tool_calls":
    print("-" * 80)
    messages.append(response.choices[0].message)
    for tool_call in response.choices[0].message.tool_calls:
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
    response = openai.chat.completions.create(
        model=model, messages=messages, tools=tools
    )

print("-" * 80)
print(response.choices[0].message.content)
