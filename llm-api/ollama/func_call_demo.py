import os, ollama


def multiply(a: int, b: int) -> int:
    return a * b


def add(a: int, b: int) -> int:
    return a + b


available_functions = {
    "multiply": multiply,
    "add": add,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "int",
                    },
                    "b": {
                        "type": "int",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "int",
                    },
                    "b": {
                        "type": "int",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
]

model = os.environ.get("OLLAMA_FC_MODEL", "mistral-nemo:12b")
messages = [{"role": "user", "content": "What is (121 * 3) + 40"}]

response = ollama.chat(model=model, messages=messages, tools=tools)

tool_calls = response["message"]["tool_calls"]
print("tool_calls:", tool_calls)

messages.append(response["message"])

for tool in tool_calls:
    function_to_call = available_functions[tool["function"]["name"]]
    function_response = function_to_call(
        tool["function"]["arguments"]["a"], tool["function"]["arguments"]["b"]
    )
    print("function_response:", function_response)
    messages.append({"role": "tool", "content": str(function_response)})

print("messages:", messages)
response = ollama.chat(model=model, messages=messages)
print(response["message"]["content"])
