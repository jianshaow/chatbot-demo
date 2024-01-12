from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    set_global_service_context,
)
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)
llm = HuggingFaceLLM(
    model_name="lmsys/vicuna-7b-v1.5",
    tokenizer_name="lmsys/vicuna-7b-v1.5",
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    device_map="auto",
)

service_context = ServiceContext.from_defaults(embed_model="local", llm=llm)
set_global_service_context(service_context)

documents = SimpleDirectoryReader("data").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(documents, show_progress=True)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
