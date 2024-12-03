from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
)

system_prompt = SystemMessage(
    "You are bad at math but are an expert at using a calculator"
)

examples = [
    HumanMessage("What is (2 + 3) * 4"),
    AIMessage(
        "",
        additional_kwargs={
            "function_call": {
                "id": "1",
                "name": "add",
                "arguments": '{"a": 2, "b": 3}',
            }
        },
        tool_calls=[ToolCall({"id": "1", "name": "add", "args": {"a": 2, "b": 3}})],
    ),
    ToolMessage("5", tool_call_id="1"),
    AIMessage(
        "",
        additional_kwargs={
            "function_call": {
                "id": "2",
                "name": "multiply",
                "arguments": '{"a": 5, "b": 4}',
            }
        },
        tool_calls=[
            ToolCall({"id": "2", "name": "multiply", "args": {"a": 5, "b": 4}})
        ],
    ),
    ToolMessage("20", tool_call_id="2"),
    AIMessage("(2 + 3) * 4 = 20."),
]
