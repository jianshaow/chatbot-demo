import torch, numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from common import hf_embed_model as model_name


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
    elif "nomic" in model_name:
        return mean_pooling
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


sentences = ["What did the author do growing up?"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.eval()
print("-" * 80)
print("embed model:", model_name)

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    model_output = model(**encoded_input)

pooling = pooling_func(model_name)
sentence_embeddings = pooling(model_output, encoded_input["attention_mask"])
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
sentence_embeddings = np.asarray([emb.numpy() for emb in sentence_embeddings]).tolist()

embedding = sentence_embeddings[0]

print("-" * 80)
print("dimension:", len(embedding))
print(embedding[:4])
print("-" * 80)
