import google.generativeai as genai

from common import gemini_fc_model as model_name
from common import gemini_few_shoted as few_shoted
from common.functions import fns
from common.prompts import fn_call_question as question
from common.prompts import fn_call_system as system_prompt
from common.prompts import gemini_examples as examples

genai.configure(transport="rest")

model_kwargs = {}
chat_kwargs = {}
if few_shoted:
    model_kwargs["system_instruction"] = system_prompt
    chat_kwargs["history"] = examples

model = genai.GenerativeModel(model_name=model_name, tools=fns.values(), **model_kwargs)
print("-" * 80)
print("fn call model:", model_name)

chat = model.start_chat(**chat_kwargs)
response = chat.send_message(question)

going = True
while going:
    print("-" * 80)

    results = []
    has_fn_call = False
    for part in response.parts:
        if fn := part.function_call:
            args = (
                "{" + ", ".join(f'"{key}": {val}' for key, val in fn.args.items()) + "}"
            )
            print("=== Calling Function ===")
            print("Calling function:", fn.name, "with args:", args)
            fn_result = fns[fn.name](**fn.args)
            print("Got output:", fn_result)
            print("========================\n")
            results.append((fn.name, fn_result))

            has_fn_call = True

    if has_fn_call:
        response_parts = genai.protos.Content(
            parts=[
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fn, response={"result": result}
                    )
                )
                for fn, result in results
            ]
        )
        response = chat.send_message(response_parts)
    else:
        going = False
        print(response.text)
