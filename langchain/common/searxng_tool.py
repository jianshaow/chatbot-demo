from langchain_community.tools.searx_search.tool import SearxSearchResults
from langchain_community.utilities import SearxSearchWrapper

from common import searxng_host

wrapper = SearxSearchWrapper(searx_host=searxng_host)

github_tool = SearxSearchResults(
    name="Github",
    wrapper=wrapper,
    kwargs={
        "engines": ["github"],
    },
)

arxiv_tool = SearxSearchResults(
    name="Arxiv", wrapper=wrapper, kwargs={"engines": ["arxiv"]}
)

if __name__ == "__main__":
    print(arxiv_tool.run("what is a large language model?"))
