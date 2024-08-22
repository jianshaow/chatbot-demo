system_prompt = {
    "role": "system",
    "content": "You are bad at math but are an expert at using a calculator",
}

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
