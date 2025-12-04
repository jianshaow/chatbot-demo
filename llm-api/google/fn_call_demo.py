from common import google_fc_model as model_name
from common import google_few_shoted as few_shoted
from common.functions import fns
from common.prompts import fn_call_adv_question as question
from common.prompts import fn_call_system as system_prompt
from common.prompts import google_examples as examples
from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    GenerateContentConfig,
    Part,
)

model_kwargs = {}
messages = []
if few_shoted:
    model_kwargs["system_instruction"] = system_prompt
    messages.extend(examples)

config = GenerateContentConfig(
    tools=list(fns.values()),
    automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
    **model_kwargs,
)
client = genai.Client()

print("-" * 80)
print("fn call model:", model_name)
chat = client.chats.create(model=model_name, config=config, history=messages)

response = chat.send_message(question)

while response.function_calls:
    print("-" * 80)

    results = []
    for fn in response.function_calls:
        if fn.args:
            args = (
                "{" + ", ".join(f'"{key}": {val}' for key, val in fn.args.items()) + "}"
            )
            print("=== Calling Function ===")
            print("Calling function:", fn.name, "with args:", args)
            if fn.name:
                fn_result = fns[fn.name](**fn.args)
                print("Got output:", fn_result)
                print("========================\n")
                results.append((fn.name, fn_result))

    fn_response_parts = [
        Part.from_function_response(
            name=fn,
            response={"result": result},
        )
        for fn, result in results
    ]
    response = chat.send_message(fn_response_parts)

if (
    response.candidates
    and response.candidates[0].content
    and response.candidates[0].content.parts
):
    print("-" * 80)
    print(response.candidates[0].content.parts[0].text)
