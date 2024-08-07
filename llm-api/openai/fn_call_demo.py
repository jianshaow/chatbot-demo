import os, json, openai


def multiply(a: int, b: int) -> int:
    return a * b


def add(a: int, b: int) -> int:
    return a + b


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

fns = {
    "multiply": multiply,
    "add": add,
}

messages = [
    {"role": "user", "content": "What is (121 * 3) + 42?"},
]

model = os.environ.get("OPENAI_FC_MODEL", "gpt-4o-mini")
response = openai.chat.completions.create(model=model, messages=messages, tools=tools)

while response.choices[0].finish_reason == "tool_calls":
    messages.append(response.choices[0].message)
    for tool_call in response.choices[0].message.tool_calls:
        fn = fns[tool_call.function.name]
        fn_args = json.loads(tool_call.function.arguments)
        fn_response = fn(**fn_args)
        messages.append(
            {
                "role": "tool",
                "content": str(fn_response),
                "tool_call_id": tool_call.id,
            }
        )
    response = openai.chat.completions.create(
        model=model, messages=messages, tools=tools
    )

print(response.choices[0].message.content)
