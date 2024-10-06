from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

from common import openai_fc_model as model
from common.fn_tools import tools

llm = OpenAI(model=model)
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

response = agent.chat("What is (121 * 3) + 42?")
print(str(response))
