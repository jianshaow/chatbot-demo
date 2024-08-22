import os

from langchain_core.messages import HumanMessage
from langchain_community.chat_models.ollama import ChatOllama

from common.fn_tools import tools
from common.functions import fns

base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
model = os.environ.get("OLLAMA_FC_MODEL", "llama3.1:8b")
llm = ChatOllama(base_url=base_url, model=model)
llm_with_tools = llm.bind_tools(tools)

messages = [HumanMessage("What is (121 * 3) + 42?")]
response = llm_with_tools.invoke(messages)

while response.tool_calls:
    messages.append(response)

    for tool_call in response.tool_calls:
        fn = fns[tool_call["name"]]
        print("=== Calling Function ===")
        print(
            "Calling function:",
            tool_call["name"],
            "with args:",
            tool_call["args"],
        )
        fn_result = fn.invoke(tool_call)
        print("Got output:", fn_result.content)
        print("========================\n")
        messages.append(fn_result)
    response = llm_with_tools.invoke(messages)

print(response.content)
