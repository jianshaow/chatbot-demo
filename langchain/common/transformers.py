from transformers import Pipeline, pipeline


def trfs_pipeline(model: str, model_kwargs: dict):
    pipe: Pipeline = pipeline(
        "text-generation",
        model=model,
        torch_dtype="auto",
        max_new_tokens=512,
        model_kwargs=model_kwargs,
    )
    if pipe.generation_config and pipe.tokenizer:
        pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
    return pipe
