from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import SearxSearchWrapper

from common import searxng_host

searxng = SearxSearchWrapper(searx_host=searxng_host, k=1)
response = searxng.run("what is a large language model?")
print(response)

print("-" * 80)
tools = load_tools(
    ["searx-search-results-json"], searx_host=searxng_host, engines=["github"], num_results=1
)
print(tools[0].run("what is stable diffusion?"))
