from transformers import pipeline

generator = pipeline("conversational")

messages = [
    {
        "role": "system",
        "content": "You are a pirate with a colorful personality.",
    },
    {"role": "user", "content": "What is your name?"},
]

response = generator(messages)

print("-" * 80)
print(response)
print("-" * 80)
