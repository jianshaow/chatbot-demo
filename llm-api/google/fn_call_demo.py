import google.generativeai as genai

from common import gemini_fc_model as model_name
from common.functions import fns

genai.configure(transport="rest")

model = genai.GenerativeModel(model_name=model_name, tools=fns.values())
print("-" * 80)
print("fn call model:", model_name)

chat = model.start_chat()
response = chat.send_message("What is (121 * 3) + 42?")

results = {}
going = True
while going:
    print("-" * 80)
    has_function_call = False
    for part in response.parts:
        if fn := part.function_call:
            has_function_call = True
            args = (
                "{" + ", ".join(f'"{key}": {val}' for key, val in fn.args.items()) + "}"
            )
            print("=== Calling Function ===")
            print("Calling function:", fn.name, "with args:", args)
            fn_result = fns[fn.name](**fn.args)
            print("Got output:", fn_result)
            print("========================\n")
            results[fn.name] = fn_result

            response_parts = genai.protos.Content(
                parts=[
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fn, response={"result": result}
                        )
                    )
                    for fn, result in results.items()
                ]
            )

    if has_function_call:
        response = chat.send_message(response_parts)
    else:
        going = False
        print(response.text)
