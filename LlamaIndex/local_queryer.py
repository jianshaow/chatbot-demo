import chromadb
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.prompts import PromptTemplate
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import HuggingFaceLLM

db = chromadb.PersistentClient(path="LlamaIndex/chroma_local")
chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="lmsys/vicuna-7b-v1.5",
    model_name="lmsys/vicuna-7b-v1.5",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
)
service_context = ServiceContext.from_defaults(embed_model="local", llm=llm)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context, service_context=service_context
)

query_engine = index.as_query_engine()
question = "What did the author do growing up?"
print("User:", question)
response = query_engine.query(question)
print("AI:", response)
