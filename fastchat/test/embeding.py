import openai

openai.api_key = "EMPTY"
openai.base_url = "http://172.17.0.1:8000/v1/"

model = "text-embedding-ada-002"

embeddings = openai.embeddings.create(
    input="What did the author do growing up?", model=model
)
print(len(embeddings.data[0].embedding))
