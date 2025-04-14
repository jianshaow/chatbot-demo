from common import get_args
from google.ai import generativelanguage as glm

chat_system = "You are a pirate with a colorful personality."
fn_call_system = "You are bad at math but are an expert at using a calculator"

embed_question = get_args(1, "What did the author do growing up?")
chat_question = get_args(1, "What is your name?")
fn_call_question = get_args(1, "What is (121 * 3) + 42?")
fn_call_adv_question = get_args(1, "What is (121 * 3) + (6 * 7)?")
mm_question = get_args(1, "Identify the city where this photo was taken.")
mm_image_url = "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"


chat_system_message = {"role": "system", "content": chat_system}
chat_question_message = {"role": "user", "content": chat_question}

fn_call_system_message = {"role": "system", "content": fn_call_system}
fn_call_question_message = {"role": "user", "content": fn_call_question}
fn_call_adv_question_message = {"role": "user", "content": fn_call_adv_question}

mm_question_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": mm_question},
        {
            "type": "image_url",
            "image_url": {"url": mm_image_url},
        },
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
    glm.Content(
        role="user",
        parts=[
            glm.Part(text="What is (2 + 3) * 4"),
        ],
    ),
    glm.Content(
        role="model",
        parts=[
            glm.Part(function_call=glm.FunctionCall(name="add", args={"a": 2, "b": 3})),
        ],
    ),
    glm.Content(
        role="user",
        parts=[
            glm.Part(
                function_response=glm.FunctionResponse(
                    name="add", response={"result": 5}
                )
            ),
        ],
    ),
    glm.Content(
        role="model",
        parts=[
            glm.Part(
                function_call=glm.FunctionCall(name="multiply", args={"a": 5, "b": 4})
            ),
        ],
    ),
    glm.Content(
        role="user",
        parts=[
            glm.Part(
                function_response=glm.FunctionResponse(
                    name="multiply", response={"result": 20}
                )
            ),
        ],
    ),
    glm.Content(
        role="model",
        parts=[
            glm.Part(text="(2 + 3) * 4 = 20."),
        ],
    ),
]
