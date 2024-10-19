from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from common import openai_fc_model as model
from common.fn_tools import tools
from common.functions import fns

llm = ChatOpenAI(model=model)
llm_with_tools = llm.bind_tools(tools)

query = "What is (121 * 3) + 42?"
messages = [HumanMessage(query)]
response = llm_with_tools.invoke(messages)

while response.response_metadata["finish_reason"] == "tool_calls":
    print("-" * 80)
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

print("-" * 80)
print(response.content)
