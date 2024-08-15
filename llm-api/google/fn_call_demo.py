import os, json, google.generativeai as genai

genai.configure(transport="rest")


def multiply(a: int, b: int) -> int:
    return a * b


def add(a: int, b: int) -> int:
    return a + b


tools = [multiply, add]
fns = {
    "multiply": multiply,
    "add": add,
}

model_name = os.environ.get("GEMINI_FC_MODEL", "models/gemini-1.5-flash")
model = genai.GenerativeModel(model_name=model_name, tools=tools)

chat = model.start_chat()
response = chat.send_message("What is (121 * 3) + 42?")

results = {}
going = True
while going:
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

            response = chat.send_message(response_parts)
        else:
            going = False
            print(response.text)
