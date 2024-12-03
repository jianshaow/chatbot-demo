import ollama

from common.functions import fns
from common.fn_tools import tools
from common.prompts import ollama_examples as examples, system_prompt
from common import ollama_fc_model as model

print("-" * 80)
print("chat model:", model)

messages = [
    system_prompt,
    *examples,
    {"role": "user", "content": "What is (121 * 3) + (6 * 7)"},
]

response = ollama.chat(model=model, messages=messages, tools=tools)

while response["message"].get("tool_calls"):
    print("-" * 80)
    messages.append(response["message"])

    tool_calls = response["message"]["tool_calls"]
    for tool in tool_calls:
        fn_name = tool["function"]["name"]
        fn = fns[fn_name]
        fn_args = tool["function"]["arguments"]
        print("=== Calling Function ===")
        print("Calling function:", fn_name, "with args:", fn_args)
        fn_result = fn(**fn_args)
        print("Got output:", fn_result)
        print("========================\n")
        messages.append({"role": "tool", "content": str(fn_result)})

    response = ollama.chat(model=model, messages=messages, tools=tools)

print("-" * 80)
print(response["message"]["content"])
