from langchain_community.utilities import SearxSearchWrapper

from common import (
    get_basic_auth_headers,
    searxng_host,
    searxng_password,
    searxng_username,
)

searxng = SearxSearchWrapper(
    searx_host=searxng_host,
    k=1,
    headers=get_basic_auth_headers(searxng_username, searxng_password),
)
response = searxng.run("what is a large language model?")
print(response)
