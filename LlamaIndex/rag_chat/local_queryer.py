import sys, torch, chromadb
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.prompts import PromptTemplate
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import HuggingFaceLLM

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
chroma_collection = db.get_or_create_collection("local")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

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

model_name = "meta-llama/Llama-2-7b-chat-hf"

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=model_name,
    model_name=model_name,
    device_map="auto",
    model_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
)
service_context = ServiceContext.from_defaults(embed_model="local", llm=llm)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context, service_context=service_context
)

query_engine = index.as_query_engine(streaming=True)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
print("User:", question, sep="\n")
response = query_engine.query(question)
print("AI:")
response.print_response_stream()
print("\n")
