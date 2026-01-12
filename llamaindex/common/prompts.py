from llama_index.core.base.llms.types import TextBlock, ToolCallBlock
from llama_index.core.llms import ChatMessage

from common import get_args

tool_call_system = "You are bad at math but are an expert at using a calculator"

system_message = ChatMessage(role="system", content=tool_call_system)

tool_call_question = get_args(1, "What is (121 * 3) + (6 * 7)?") or ""

question_message = ChatMessage(role="user", content=tool_call_question)

examples = [
    ChatMessage(
        role="user",
        blocks=[
            TextBlock(block_type="text", text="What is (2 + 3) * 4"),
        ],
    ),
    ChatMessage(
        role="assistant",
        blocks=[
            ToolCallBlock(
                block_type="tool_call",
                tool_call_id="call_H8CcFDO4O51jD5aJBp7N3eKh",
                tool_name="add",
                tool_kwargs={"a": 2, "b": 3},
            ),
        ],
    ),
    ChatMessage(
        role="tool",
        blocks=[
            TextBlock(block_type="text", text="5"),
        ],
    ),
    ChatMessage(
        role="assistant",
        blocks=[
            ToolCallBlock(
                block_type="tool_call",
                tool_call_id="call_C1nzoLmWboI6SvIwzp57dvHe",
                tool_name="multiply",
                tool_kwargs={"a": 5, "b": 4},
            ),
        ],
    ),
    ChatMessage(
        role="tool",
        blocks=[
            TextBlock(block_type="text", text="20"),
        ],
    ),
    ChatMessage(
        role="assistant",
        blocks=[
            TextBlock(block_type="text", text="(2 + 3) * 4 = 20."),
        ],
    ),
]
