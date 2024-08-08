import os, ollama


def multiply(a: int, b: int) -> int:
    print("multiply:", a, b)
    return a * b


def add(a: int, b: int) -> int:
    print("add:", a, b)
    return a + b


fns = {
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
messages = [{"role": "user", "content": "What is (121 * 3) + 42"}]

response = ollama.chat(model=model, messages=messages, tools=tools)

tool_calls = response["message"]["tool_calls"]
print("tool_calls:", tool_calls)

messages.append(response["message"])

for tool in tool_calls:
    fn = fns[tool["function"]["name"]]
    fn_args = tool["function"]["arguments"]
    fn_response = fn(**fn_args)
    print("fn_response:", fn_response)
    messages.append({"role": "tool", "content": str(fn_response)})

print("messages:", messages)
response = ollama.chat(model=model, messages=messages)
print(response["message"]["content"])
