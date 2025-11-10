from agents import Agent, Runner

from common import openai_chat_model as model

agent = Agent(model=model, name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
