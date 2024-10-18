from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

from common import ollama_base_url as base_url, ollama_fc_model as model
from common.fn_tools import tools
from common.functions import fns

llm = Ollama(base_url=base_url, model=model)

messages = [ChatMessage(role="user", content="What is (121 * 3) + 42?")]
response = llm.chat_with_tools(tools, messages[0])

while response.message.additional_kwargs.get("tool_calls"):
    print("-" * 80)
    messages.append(response.message)
    for tool_call in response.message.additional_kwargs.get("tool_calls"):
        fn_name = tool_call["function"]["name"]
        fn = fns[fn_name]
        fn_args = tool_call["function"]["arguments"]
        print("=== Calling Function ===")
        print(
            "Calling function:",
            fn_name,
            "with args:",
            fn_args,
        )
        fn_result = fn(**fn_args)
        print("Got output:", fn_result)
        print("========================\n")
        tool_message = ChatMessage(
            content=str(fn_result),
            role="tool",
            additional_kwargs={"name": fn_name},
        )
        messages.append(tool_message)
    response = llm.chat_with_tools(tools, chat_history=messages)

print(response.message.content)
