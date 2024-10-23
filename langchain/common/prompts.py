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
    HumanMessage("What is (2 * 3) + 4"),
    AIMessage(
        "",
        tool_calls=[
            ToolCall({"id": "1", "name": "multiply", "args": {"a": 2, "b": 3}})
        ],
    ),
    ToolMessage("6", tool_call_id="1"),
    AIMessage(
        "",
        tool_calls=[ToolCall({"id": "2", "name": "add", "args": {"a": 6, "b": 4}})],
    ),
    ToolMessage("10", tool_call_id="2"),
    AIMessage("(2 * 3) + 4 = 10."),
]
