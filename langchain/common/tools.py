from langchain.tools import tool
from langchain_core.vectorstores import VectorStore

from common.calc_func import add, multiply

calc_tools = [add, multiply]


def get_retrieve_tool(vector_store: VectorStore):

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context
