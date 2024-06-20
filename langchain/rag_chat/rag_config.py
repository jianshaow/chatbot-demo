import os, sys
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from typing import Type

DATA_PATH = "data"
DATA_PATH_EN = "data_en"
DATA_PATH_ZH = "data_zh"

DEFAULT_QUESTION = "What did the author do growing up?"
DEFAULT_QUESTION_EN = "Why the old man go fishing?"
DEFAULT_QUESTION_ZH = "地球发动机都安装在哪里？"


class RagChatConfig:
    def __init__(
        self,
        embedding_model: Type[Embeddings],
        embedding_model_name: str,
        chat_model: Type[BaseLanguageModel],
        chat_model_name: str,
        bnb_quantized: bool = True,
        data_path: str = DATA_PATH,
        vector_db_collection: str = "hface",
        defalut_question: str = DEFAULT_QUESTION,
    ):
        self.__embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.__chat_model = chat_model
        self.chat_model_name = chat_model_name
        self.bnb_quantized = bnb_quantized
        self.data_path = data_path
        self.vector_db_path = os.environ.get("CHROMA_DB_DIR", "chroma")
        self.vector_db_collection = vector_db_collection
        self.defalut_question = defalut_question

    def embedding_model(self):
        if self.__embedding_model == OllamaEmbeddings:
            base_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
            )
            return self.__embedding_model(
                base_url=base_url, model=self.embedding_model_name
            )
        return self.__embedding_model(model=self.embedding_model_name)

    def chat_model(self):
        if self.__chat_model == ChatOllama:
            base_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
            )
            return self.__chat_model(base_url=base_url, model=self.chat_model_name)
        return self.__chat_model(model=self.chat_model_name)

    def get_question(self):
        if len(sys.argv) >= 3:
            return sys.argv[2]
        else:
            return self.defalut_question


def __openai_config(
    embeddding_model_name="text-embedding-ada-002",
    chat_model_name="gpt-3.5-turbo",
    data_path=DATA_PATH,
    vector_db_collection="openai",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        OpenAIEmbeddings,
        embeddding_model_name,
        ChatOpenAI,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __gemini_config(
    embeddding_model_name="models/embedding-001",
    chat_model_name="models/gemini-1.5-flash",
    data_path=DATA_PATH,
    vector_db_collection="gemini",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        GoogleGenerativeAIEmbeddings,
        embeddding_model_name,
        ChatGoogleGenerativeAI,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __ollama_config(
    embeddding_model_name="nomic-embed-text:v1.5",
    chat_model_name="llama3:8b",
    data_path=DATA_PATH,
    vector_db_collection="ollama",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        OllamaEmbeddings,
        embeddding_model_name,
        ChatOllama,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


__config_dict = {
    "openai": __openai_config(),
    "openai_en": __openai_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="openai_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "openai_zh": __openai_config(
        data_path=DATA_PATH_ZH,
        vector_db_collection="openai_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "gemini": __gemini_config(),
    "gemini_en": __gemini_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="gemini_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "gemini_zh": __gemini_config(
        data_path=DATA_PATH_ZH,
        vector_db_collection="gemini_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "ollama": __ollama_config(),
    "ollama_en": __ollama_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="ollama_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "ollama_zh": __ollama_config(
        data_path=DATA_PATH_ZH,
        embeddding_model_name="nomic-embed-text:v1.5",
        vector_db_collection="ollama_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
}


def get_config(name="gemini"):
    if len(sys.argv) >= 2:
        return __config_dict[sys.argv[1]]
    else:
        return __config_dict[name]


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        print(vars(__config_dict[sys.argv[1]]))
    else:
        for config in __config_dict:
            print(config)
