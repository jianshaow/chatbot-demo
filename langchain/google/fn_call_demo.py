from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from common import google_fc_model as model
from common.fn_tools import tools
from common.functions import fns

llm = ChatGoogleGenerativeAI(model=model, transport="rest")
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
