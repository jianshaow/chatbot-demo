from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from common import get_args

CHAT_SYSTEM = "You are a pirate with a colorful personality."
FN_CALL_SYSTEM = "You are bad at math but are an expert at using a calculator"

embed_question = get_args(1, "What did the author do growing up?")
chat_question = get_args(1, "What is your name?")
fn_call_question = get_args(1, "What is (121 * 3) + 42?")
fn_call_adv_question = get_args(1, "What is (121 * 3) + (6 * 7)?")
mm_question1 = get_args(1, "Identify the city where this photo was taken.")
mm_question2 = get_args(1, "Give me more context for this image.")
mm_image_url = "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"

fn_call_system_message = SystemMessage(FN_CALL_SYSTEM)
fn_question_message = HumanMessage(fn_call_question)
fn_adv_question_message = HumanMessage(fn_call_adv_question)

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
    ToolMessage("5", name="add", tool_call_id="1"),
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
    ToolMessage("20", name="multiply", tool_call_id="2"),
    AIMessage("(2 + 3) * 4 = 20."),
]
