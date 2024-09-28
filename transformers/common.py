import os, torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

load_dotenv()

hf_embed_model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en")
hf_chat_model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

def new_model(
    model_name: str, bnb_enabled=os.environ.get("BNB_ENABLED", "false") == "true"
) -> PreTrainedModel:
    model_args = {}
    if bnb_enabled:
        model_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_args,
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
    model_name = os.environ.get("HF_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    model = new_model(model_name)
    tokenizer = new_tokenizer(model_name)
    print("-" * 80)
    print(generate(model, tokenizer, "who are you?"))
    print("-" * 80)
