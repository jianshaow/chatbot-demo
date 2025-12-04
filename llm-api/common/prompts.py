from common import get_args
from google.genai import types
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

chat_system = "You are a pirate with a colorful personality."
fn_call_system = "You are bad at math but are an expert at using a calculator"

embed_question = get_args(1, "What did the author do growing up?")
chat_question = get_args(1, "What is your name?")
fn_call_question = get_args(1, "What is (121 * 3) + 42?")
fn_call_adv_question = get_args(1, "What is (121 * 3) + (6 * 7)?")
mm_question = get_args(1, "Identify the city where this photo was taken.")
mm_image_url = "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"

chat_system_message: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": chat_system,
}
chat_question_message: ChatCompletionUserMessageParam = {
    "role": "user",
    "content": chat_question,
}

fn_call_system_message: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": fn_call_system,
}
fn_call_question_message: ChatCompletionUserMessageParam = {
    "role": "user",
    "content": fn_call_question,
}
fn_call_adv_question_message: ChatCompletionUserMessageParam = {
    "role": "user",
    "content": fn_call_adv_question,
}

mm_question_message: ChatCompletionUserMessageParam = {
    "role": "user",
    "content": [
        {
            "type": "image_url",
            "image_url": {"url": mm_image_url},
        },
        {"type": "text", "text": mm_question},
    ],
}

ollama_examples = [
    {
        "role": "user",
        "content": "What is (2 + 3) * 4",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "add", "arguments": {"a": 2, "b": 3}}}],
    },
    {
        "role": "tool",
        "content": "5",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "multiply", "arguments": {"a": 5, "b": 4}}}
        ],
    },
    {
        "role": "tool",
        "content": "20",
    },
    {
        "role": "assistant",
        "content": "(2 + 3) * 4 = 20.",
    },
]

google_examples = [
    types.Content(
        role="user",
        parts=[
            types.Part(text="What is (2 + 3) * 4"),
        ],
    ),
    types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="add", args={"a": 2, "b": 3})
            ),
        ],
    ),
    types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="add", response={"result": 5}
                )
            ),
        ],
    ),
    types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="multiply", args={"a": 5, "b": 4})
            ),
        ],
    ),
    types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="multiply", response={"result": 20}
                )
            ),
        ],
    ),
    types.Content(
        role="model",
        parts=[
            types.Part(text="(2 + 3) * 4 = 20."),
        ],
    ),
]
