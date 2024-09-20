import os
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

from common.fn_tools import tools

model = os.environ.get("OPENAI_FC_MODEL", "gpt-4o-mini")
llm = OpenAI(model=model)
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

response = agent.chat("What is (121 * 3) + 42?")
print(str(response))
