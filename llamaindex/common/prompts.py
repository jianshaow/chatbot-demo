import sys
from llama_index.core.llms import ChatMessage

tool_call_system = "You are bad at math but are an expert at using a calculator"

system_message = ChatMessage(role="system", content=tool_call_system)

tool_call_question = (
    len(sys.argv) == 2 and sys.argv[1] or "What is (121 * 3) + 42?"
)

question_message = ChatMessage(role="user", content=tool_call_question)

examples = [
    ChatMessage(
        role="user",
        content="What is (2 + 3) * 4",
    ),
    ChatMessage(
        role="assistant",
        content="",
        additional_kwargs={
            "tool_calls": [{"function": {"name": "add", "arguments": {"a": 2, "b": 3}}}]
        },
    ),
    ChatMessage(
        role="tool",
        content="5",
    ),
    ChatMessage(
        role="assistant",
        content="",
        additional_kwargs={
            "tool_calls": [
                {"function": {"name": "multiply", "arguments": {"a": 5, "b": 4}}}
            ]
        },
    ),
    ChatMessage(
        role="tool",
        content="20",
    ),
    ChatMessage(
        role="assistant",
        content="(2 + 3) * 4 = 20.",
    ),
]
