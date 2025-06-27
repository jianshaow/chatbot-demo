import torch
import torch.nn.functional as F
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from common import hf_embed_zh_model as embed_model


def cls_pooling(model_output, *args):
    return model_output[0][:, 0]


def mean_pooling(model_output, attention_mask: torch.Tensor):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def pooling_func(model_name: str):
    if "bge" in model_name:
        return cls_pooling
    elif "nomic" in model_name or "paraphrase" in model_name:
        return mean_pooling
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def main():
    sentences = ["地球发动机都安装在哪里？"]

    tokenizer = AutoTokenizer.from_pretrained(embed_model)
    model = AutoModel.from_pretrained(embed_model, trust_remote_code=True)
    model.eval()
    print("-" * 80)
    print("embed model:", embed_model)

    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    pooling = pooling_func(embed_model)
    embeddings = pooling(model_output, encoded_input["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    embedding = embeddings[0].tolist()

    print("-" * 80)
    print("dimension:", len(embedding))
    print(embedding[:4])
    print("-" * 80)


if __name__ == "__main__":
    main()
