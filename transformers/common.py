import os
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

load_dotenv()

hf_embed_model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en")
hf_chat_model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-3B-Instruct")


def default_model_kwargs() -> dict[str, str]:
    model_kwargs = {}
    bnb_enabled = os.environ.get("BNB_ENABLED", "false") == "true"
    if bnb_enabled:
        import torch
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    return model_kwargs


def new_model(model_name: str, model_kwargs=None) -> PreTrainedModel:
    if model_kwargs is None:
        model_kwargs = default_model_kwargs()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
        device_map="auto",
    )
    return model


def new_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def generate(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str):

    input_ids = tokenizer(
        prompt, return_tensors="pt", return_attention_mask=False
    ).input_ids.to(model.device)

    generation_config = GenerationConfig.from_pretrained(
        model.name_or_path, max_length=1024
    )
    tokens = model.generate(input_ids, generation_config=generation_config)

    token_ids = tokens[0][input_ids.size(1) :]
    response = tokenizer.decode(token_ids, skip_special_tokens=True)

    return response


if __name__ == "__main__":
    model = new_model(hf_chat_model)
    tokenizer = new_tokenizer(hf_chat_model)
    print("-" * 80)
    print(generate(model, tokenizer, "who are you?"))
    print("-" * 80)
