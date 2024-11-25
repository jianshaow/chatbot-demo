import torch, numpy as np
from transformers import AutoTokenizer, AutoModel

from common import hf_embed_model as model_name

question = "What did the author do growing up?"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

encoded_input = tokenizer(
    [question], padding=True, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    model_output = model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
sentence_embeddings = np.asarray([emb.numpy() for emb in sentence_embeddings]).tolist()

embeddings = sentence_embeddings[0]
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80)
