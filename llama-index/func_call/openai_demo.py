from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool


def multiply(a: int, b: int) -> int:
    return a * b


def add(a: int, b: int) -> int:
    return a + b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

llm = OpenAI()
agent = OpenAIAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is (121 * 3) + 42?")
print(str(response))
