import sys, torch
from llama_index.embeddings import BaseEmbedding, HuggingFaceEmbedding, OpenAIEmbedding
from llama_index.llms import LLM, HuggingFaceLLM, OpenAI
from llama_index.prompts import PromptTemplate

DATA_PATH = "LlamaIndex/data"
DATA_PATH_ZH = "LlamaIndex/data_zh"


class RagChatConfig:
    def __init__(
        self,
        embedding_model: type(BaseEmbedding),
        embedding_model_name: str,
        chat_model: type(LLM),
        chat_model_name: str,
        data_path: str = DATA_PATH,
        vector_db_path: str = "LlamaIndex/chroma_db",
        vector_db_collection: str = "local",
        defalut_question: str = "What did the author do growing up?",
    ):
        self.__embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.__chat_model = chat_model
        self.chat_model_name = chat_model_name
        self.data_path = data_path
        self.vector_db_path = vector_db_path
        self.vector_db_collection = vector_db_collection
        self.defalut_question = defalut_question

    def embedding_model(self):
        return self.__embedding_model(model=self.embedding_model_name)

    def chat_model(self):
        if self.__chat_model == type(HuggingFaceLLM):
            return __hf_chat_model(self.chat_model_name)
        else:
            return self.__chat_model(self.chat_model_name)

    def get_question(self):
        if len(sys.argv) >= 3:
            return sys.argv[2]
        else:
            return self.defalut_question


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


def __hf_chat_model(model_name="meta-llama/Llama-2-7b-chat-hf"):
    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=model_name,
        model_name=model_name,
        device_map="auto",
        model_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
    )


def __openai_config(
    embeddding_model_name="text-embedding-ada-002",
    chat_model_name="gpt-3.5-turbo",
    vector_db_collection="openai",
    data_path=DATA_PATH,
    defalut_question="What did the author do growing up?",
):
    return RagChatConfig(
        OpenAIEmbedding,
        embeddding_model_name,
        OpenAI,
        chat_model_name,
        vector_db_collection=vector_db_collection,
        data_path=data_path,
        defalut_question=defalut_question,
    )


def __hf_config(
    embeddding_model_name=None,
    chat_model_name="meta-llama/Llama-2-7b-chat-hf",
    vector_db_collection="local",
    data_path=DATA_PATH,
    defalut_question="What did the author do growing up?",
):
    return RagChatConfig(
        HuggingFaceEmbedding,
        embeddding_model_name,
        HuggingFaceLLM,
        chat_model_name,
        vector_db_collection=vector_db_collection,
        data_path=data_path,
        defalut_question=defalut_question,
    )


HYBRID = RagChatConfig(
    HuggingFaceEmbedding,
    "BAAI/bge-large-en-v1.5",
    OpenAI,
    "gpt-3.5-turbo",
)

HYBRID_ZH = RagChatConfig(
    HuggingFaceEmbedding,
    "BAAI/bge-large-zh-v1.5",
    OpenAI,
    "gpt-3.5-turbo",
    vector_db_collection="local_zh",
)
__config_dict = {
    "openai": __openai_config(),
    "openai_zh": __openai_config(
        vector_db_collection="openai_zh",
        data_path=DATA_PATH_ZH,
        defalut_question="杨志是个怎样的人?",
    ),
    "local": __hf_config(),
    "local_zh": __hf_config(
        vector_db_collection="local_zh",
        data_path=DATA_PATH_ZH,
        defalut_question="杨志是个怎样的人?",
    ),
    "hybrid": HYBRID,
    "hybrid_zh": HYBRID_ZH,
}


def get_config(name="openai"):
    if len(sys.argv) >= 2:
        return __config_dict[sys.argv[1]]
    else:
        return __config_dict[name]
