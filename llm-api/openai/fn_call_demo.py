import os, json, openai

from common.functions import fns
from common.fn_tools import tools


messages = [
    {"role": "user", "content": "What is (121 * 3) + 42?"},
]

model = os.environ.get("OPENAI_FC_MODEL", "gpt-4o-mini")
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

print(response.choices[0].message.content)
