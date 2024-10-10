from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from common import bnb_enabled, hf_chat_model


def default_model_kwargs() -> dict[str, str]:
    model_kwargs = {}

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
        torch_dtype="auto",
        device_map="auto",
        **model_kwargs,
    )
    return model


def new_tokenizer(model_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def generate(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, streaming=False
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generate_kwargs = {"max_new_tokens": 512}
    if streaming:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        generate_kwargs["streamer"] = streamer

    outputs = model.generate(**inputs, **generate_kwargs)

    if streaming:
        return streamer
    else:
        token_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        return tokenizer.batch_decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]


if __name__ == "__main__":
    model = new_model(hf_chat_model)
    tokenizer = new_tokenizer(hf_chat_model)
    print("-" * 80)
    print(generate(model, tokenizer, "who are you?"))
    print("-" * 80)
