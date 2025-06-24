from common import google_fc_model as model_name
from common import google_few_shoted as few_shoted
from common.functions import fns
from common.prompts import fn_call_adv_question as question
from common.prompts import fn_call_system as system_prompt
from common.prompts import google_examples as examples
from google import genai
from google.genai import types

model_kwargs = {}
messages = []
if few_shoted:
    model_kwargs["system_instruction"] = system_prompt
    messages.extend(examples)

config = types.GenerateContentConfig(
    tools=list(fns.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    **model_kwargs,
)
client = genai.Client()

print("-" * 80)
print("fn call model:", model_name)
chat = client.chats.create(config=config, model=model_name)

messages.append(question)
response = chat.send_message(messages)

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
        types.Part.from_function_response(
            name=fn,
            response={"result": result},
        )
        for fn, result in results
    ]
    response = chat.send_message(fn_response_parts)  # type: ignore

print(response.text)
