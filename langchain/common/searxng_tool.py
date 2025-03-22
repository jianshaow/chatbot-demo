from langchain_community.tools.searx_search.tool import SearxSearchResults
from langchain_community.utilities import SearxSearchWrapper

from common import (
    get_basic_auth_headers,
    searxng_host,
    searxng_password,
    searxng_username,
)

wrapper = SearxSearchWrapper(
    searx_host=searxng_host,
    headers=get_basic_auth_headers(searxng_username, searxng_password),
)

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
    print(github_tool.run("stable diffusion"))
    print("-" * 80)
    print(arxiv_tool.run("2201.11903"))
