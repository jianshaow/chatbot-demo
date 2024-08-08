import os, ollama

model_name = os.environ.get("OLLAMA_CHAT_MODEL", "vicuna:13b")
print("-" * 80)
print("chat model:", model_name)
response = ollama.chat(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a pirate with a colorful personality."},
        {"role": "user", "content": "What is your name?"},
    ],
    stream=True,
)

print("-" * 80)
for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)
print("\n", "-" * 80, sep="")
