import openai

openai.api_key = "EMPTY"
openai.base_url = "http://172.17.0.1:8000/v1/"

model = "gpt-3.5-turbo"
prompt = "Once upon a time"

completion = openai.completions.create(model=model, prompt=prompt, max_tokens=64)
print(prompt + completion.choices[0].text)

completion = openai.chat.completions.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
print(completion.choices[0].message.content)
