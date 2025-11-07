import os
from typing import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from common import (
    data_base_dir,
    db_base_dir,
    get_args,
    google_chat_model,
    google_embed_model,
    hf_chat_model,
    hf_embed_model,
    ollama_chat_model,
    ollama_embed_model,
    openai_chat_model,
    openai_embed_model,
)
from common.google import get_chat_model as get_google_chat_model
from common.google import get_embed_model as get_google_embed_model
from common.hgface import get_chat_model as get_hf_chat_model
from common.hgface import get_embed_model as get_hf_embed_model
from common.ollama import get_chat_model as get_ollama_chat_model
from common.ollama import get_embed_model as get_ollama_embed_model
from common.openai import get_chat_model as get_openai_chat_model
from common.openai import get_embed_model as get_openai_embed_model

DEFAULT_DATA = "default"
DATA_EN = "en-text"
DATA_ZH = "zh-text"

DEFAULT_QUESTION = "What did the author do growing up?"
DEFAULT_QUESTION_EN = "Why the old man go fishing?"
DEFAULT_QUESTION_ZH = "地球发动机都安装在哪里？"


class RagChatConfig:

    def __init__(
        self,
        name: str,
        get_embed_model: Callable[[str], Embeddings],
        embed_model_name: str,
        get_chat_model: Callable[[str], BaseChatModel],
        chat_model_name: str,
        data_dir: str = DEFAULT_DATA,
        defalut_question: str = DEFAULT_QUESTION,
    ):
        self.name = name
        self.get_embed_model = get_embed_model
        self.embed_model_name = embed_model_name
        self.get_chat_model = get_chat_model
        self.chat_model_name = chat_model_name
        self.__data_dir = data_dir
        self.db_base_dir = db_base_dir
        self.data_base_dir = data_base_dir
        self.defalut_question = defalut_question

    @property
    def vector_db_path(self):
        return os.path.join(self.db_base_dir, self.name)

    @property
    def data_dir(self):
        return os.path.join(self.data_base_dir, self.__data_dir)

    @property
    def vector_db_collection(self):
        escaped = self.embed_model_name.replace(":", "_").replace("/", "_")
        return self.__data_dir + "__" + escaped

    def embed_model(self) -> Embeddings:
        return self.get_embed_model(self.embed_model_name)

    def chat_model(self) -> BaseChatModel:
        return self.get_chat_model(self.chat_model_name)

    def get_question(self):
        return get_args(2, self.defalut_question)


def __openai_config(
    data_dir=DEFAULT_DATA,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "openai",
        get_openai_embed_model,
        openai_embed_model,
        get_openai_chat_model,
        openai_chat_model,
        data_dir=data_dir,
        defalut_question=defalut_question,
    )


def __google_config(
    data_dir=DEFAULT_DATA,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "google",
        get_google_embed_model,
        google_embed_model,
        get_google_chat_model,
        google_chat_model,
        data_dir=data_dir,
        defalut_question=defalut_question,
    )


def __ollama_config(
    embed_model_name=ollama_embed_model,
    chat_model_name=ollama_chat_model,
    data_dir=DEFAULT_DATA,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "ollama",
        get_ollama_embed_model,
        embed_model_name,
        get_ollama_chat_model,
        chat_model_name,
        data_dir=data_dir,
        defalut_question=defalut_question,
    )


def __hf_config(
    embed_model_name=hf_embed_model,
    chat_model_name=hf_chat_model,
    data_dir=DEFAULT_DATA,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "hface",
        get_hf_embed_model,
        embed_model_name,
        get_hf_chat_model,
        chat_model_name,
        data_dir=data_dir,
        defalut_question=defalut_question,
    )


__config_dict = {
    "openai": __openai_config(),
    "openai_en": __openai_config(
        data_dir=DATA_EN,
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "openai_zh": __openai_config(
        data_dir=DATA_ZH,
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "google": __google_config(),
    "google_en": __google_config(
        data_dir=DATA_EN,
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "google_zh": __google_config(
        data_dir=DATA_ZH,
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "ollama": __ollama_config(),
    "ollama_en": __ollama_config(
        data_dir=DATA_EN,
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "ollama_zh": __ollama_config(
        data_dir=DATA_ZH,
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "hf": __hf_config(),
    "hf_en": __hf_config(
        data_dir=DATA_EN,
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "hf_zh": __hf_config(
        data_dir=DATA_ZH,
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
}


def get_config():
    return __config_dict[get_args(1, "google")]


if __name__ == "__main__":
    if config_key := get_args(1, ""):
        print(vars(__config_dict[config_key]))
    else:
        for config in __config_dict:
            print(config)
