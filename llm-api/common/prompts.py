import sys
import google.generativeai as genai

tool_call_system = "You are bad at math but are an expert at using a calculator"

tool_call_question = (
    len(sys.argv) == 2 and sys.argv[1] or "What is (121 * 3) + (6 * 7)?"
)

system_message = {
    "role": "system",
    "content": tool_call_system,
}

question_message = {"role": "user", "content": tool_call_question}

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

gemini_examples = [
    genai.protos.Content(
        role="user",
        parts=[
            genai.protos.Part(text="What is (2 + 3) * 4"),
        ],
    ),
    genai.protos.Content(
        role="model",
        parts=[
            genai.protos.Part(
                function_call=genai.protos.FunctionCall(
                    name="add", args={"a": 2, "b": 3}
                )
            ),
        ],
    ),
    genai.protos.Content(
        role="user",
        parts=[
            genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name="add", response={"result": 5}
                )
            ),
        ],
    ),
    genai.protos.Content(
        role="model",
        parts=[
            genai.protos.Part(
                function_call=genai.protos.FunctionCall(
                    name="multiply", args={"a": 5, "b": 4}
                )
            ),
        ],
    ),
    genai.protos.Content(
        role="user",
        parts=[
            genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name="multiply", response={"result": 20}
                )
            ),
        ],
    ),
    genai.protos.Content(
        role="model",
        parts=[
            genai.protos.Part(text="(2 + 3) * 4 = 20."),
        ],
    ),
]
