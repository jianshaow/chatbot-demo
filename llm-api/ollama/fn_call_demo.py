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
                        "type": "integer",
                    },
                    "b": {
                        "type": "integer",
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
                        "type": "integer",
                    },
                    "b": {
                        "type": "integer",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
]

examples = [
    {
        "role": "user",
        "content": "What is (2 * 3) + 4",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "multiply", "arguments": {"a": 2, "b": 3}}}
        ],
    },
    {
        "role": "tool",
        "content": "6",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "add", "arguments": {"a": 6, "b": 4}}}],
    },
    {
        "role": "tool",
        "content": "10",
    },
    {
        "role": "assistant",
        "content": "(2 * 3) + 4 = 10.",
    },
]

messages = [
    {
        "role": "system",
        "content": "You are bad at math but are an expert at using a calculator",
    },
    *examples,
    {"role": "user", "content": "What is (121 * 3) + 42"},
]

model = os.environ.get("OLLAMA_FC_MODEL", "llama3.1:8b")
response = ollama.chat(model=model, messages=messages, tools=tools)


while response["message"].get("tool_calls"):
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

print(response["message"]["content"])
