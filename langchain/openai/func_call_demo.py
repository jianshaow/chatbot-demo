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

llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)

query = "What is (121 * 3) + 42?"
messages = [HumanMessage(query)]
response = llm_with_tools.invoke(messages)
messages.append(response)

print(response.tool_calls)
for tool_call in response.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"]]
    tool_result = selected_tool.invoke(tool_call)
    messages.append(tool_result)


response = llm_with_tools.invoke(messages)
print(response.content)
