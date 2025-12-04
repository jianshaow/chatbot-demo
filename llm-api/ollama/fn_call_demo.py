from common import ollama_fc_model as model
from common import think
from common.fn_tools import tools
from common.functions import fns
from common.prompts import fn_call_adv_question_message as question
from common.prompts import fn_call_system_message as system_prompt
from common.prompts import ollama_examples as examples
from ollama import Client

print("-" * 80)
print("fn call model:", model)

client = Client()

messages = [system_prompt, *examples, question]

response = client.chat(model=model, messages=messages, think=think, tools=tools)

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

    response = client.chat(model=model, messages=messages, think=think, tools=tools)

print("-" * 80)
print(response["message"]["content"])
