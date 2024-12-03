from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

from common import openai_fc_model as model
from common.fn_tools import tools
from common.prompts import tool_call_question as question

llm = OpenAI(model=model)
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

response = agent.chat(question)
print("-" * 80)
print(str(response))
