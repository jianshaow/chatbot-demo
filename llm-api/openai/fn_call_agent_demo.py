from agents import Agent, Runner

from common import openai_chat_model as model
from common.fn_tools import calc_tools
from common.prompts import fn_call_adv_question as question

agent = Agent(
    model=model,
    name="Assistant",
    instructions="You are a helpful assistant",
    tools=calc_tools,
)

result = Runner.run_sync(agent, question)
print(result.final_output)
