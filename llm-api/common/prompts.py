import google.generativeai as genai

system_prompt_str = "You are bad at math but are an expert at using a calculator"

system_prompt = {
    "role": "system",
    "content": system_prompt_str,
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
