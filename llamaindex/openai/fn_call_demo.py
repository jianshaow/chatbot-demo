import os, json
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from common.fn_tools import tools
from common.functions import fns

model = os.environ.get("OPENAI_FC_MODEL", "gpt-4o-mini")
llm = OpenAI(model=model)

messages = [ChatMessage(role="user", content="What is (121 * 3) + 42?")]
response = llm.chat_with_tools(tools, messages[0])

while response.message.additional_kwargs.get("tool_calls"):
    messages.append(response.message)
    for tool_call in response.message.additional_kwargs.get("tool_calls"):
        fn_name = tool_call.function.name
        fn = fns[fn_name]
        fn_args = json.loads(tool_call.function.arguments)
        print("=== Calling Function ===")
        print(
            "Calling function:",
            fn_name,
            "with args:",
            tool_call.function.arguments,
        )
        fn_result = fn(**fn_args)
        print("Got output:", fn_result)
        print("========================\n")
        tool_message = ChatMessage(
            content=str(fn_result),
            role="tool",
            additional_kwargs={
                "name": fn_name,
                "tool_call_id": tool_call.id,
            },
        )
        messages.append(tool_message)
    response = llm.chat_with_tools(tools, chat_history=messages)

print(response.message.content)
