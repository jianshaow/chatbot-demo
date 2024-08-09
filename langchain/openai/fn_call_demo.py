import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

available_tools = {"add": add, "multiply": multiply}

model = os.environ.get("OPENAI_FC_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=model)
llm_with_tools = llm.bind_tools(tools)

query = "What is (121 * 3) + 42?"
messages = [HumanMessage(query)]
response = llm_with_tools.invoke(messages)

while response.response_metadata["finish_reason"] == "tool_calls":
    messages.append(response)

    for tool_call in response.tool_calls:
        selected_tool = available_tools[tool_call["name"]]
        tool_result = selected_tool.invoke(tool_call)
        messages.append(tool_result)
    response = llm_with_tools.invoke(messages)

print(response.content)
