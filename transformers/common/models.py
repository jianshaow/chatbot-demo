from threading import Thread

from transformers.generation.streamers import TextIteratorStreamer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from common import bnb_enabled, hf_chat_model


def default_model_kwargs() -> dict[str, str]:
    model_kwargs = {}

    if bnb_enabled:
        import torch
        from transformers.utils.quantization_config import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    return model_kwargs


def new_model(model_name: str, model_kwargs=None):
    model_kwargs = default_model_kwargs() if model_kwargs is None else model_kwargs
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def new_multi_modal_model(model_name: str, model_kwargs=None):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    architecture = config.architectures[0]
    AutoModelClass = AutoModelForCausalLM

    if "Qwen2VLForConditionalGeneration" in architecture:
        AutoModelClass = Qwen2VLForConditionalGeneration
    if "Qwen2_5_VLForConditionalGeneration" in architecture:
        AutoModelClass = Qwen2_5_VLForConditionalGeneration
    if "PaliGemmaForConditionalGeneration" in architecture:
        AutoModelClass = PaliGemmaForConditionalGeneration
    if "MllamaForConditionalGeneration" in architecture:
        AutoModelClass = MllamaForConditionalGeneration

    if model_kwargs is None:
        model_kwargs = default_model_kwargs()
    model_kwargs = {
        **model_kwargs,
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }
    model = AutoModelClass.from_pretrained(model_name, **model_kwargs)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor, config


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompt: str,
    streaming=False,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = {
        **inputs,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 512,
    }

    if streaming:
        streamer = TextIteratorStreamer(
            tokenizer,  # type: ignore
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        generation_kwargs = {**generation_kwargs, "streamer": streamer}

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer
    else:
        outputs = model.generate(**generation_kwargs)
        token_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        return tokenizer.batch_decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]


def image_text_to_text(model: PreTrainedModel, processor, images, text):
    inputs = processor(text=text, images=images, padding=True, return_tensors="pt").to(
        model.device
    )
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main():
    model, tokenizer = new_model(hf_chat_model)
    response = generate(model, tokenizer, "who are you?", streaming=True)
    print("-" * 80)
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n", "-" * 80, sep="")


if __name__ == "__main__":
    main()
