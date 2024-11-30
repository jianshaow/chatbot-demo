from llama_index.core.llms import ChatMessage

system_prompt = ChatMessage(
    role="system", content="You are bad at math but are an expert at using a calculator"
)

examples = [
    ChatMessage(
        role="user",
        content="What is (2 * 3) + 4",
    ),
    ChatMessage(
        role="assistant",
        content="",
        additional_kwargs={
            "tool_calls": [
                {"function": {"name": "multiply", "arguments": {"a": 2, "b": 3}}}
            ]
        },
    ),
    ChatMessage(
        role="tool",
        content="6",
    ),
    ChatMessage(
        role="assistant",
        content="",
        additional_kwargs={
            "tool_calls": [{"function": {"name": "add", "arguments": {"a": 6, "b": 4}}}]
        },
    ),
    ChatMessage(
        role="tool",
        content="10",
    ),
    ChatMessage(
        role="assistant",
        content="(2 * 3) + 4 = 10.",
    ),
]
