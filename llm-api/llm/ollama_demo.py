import os, ollama

model = os.environ.get("OLLAMA_MODEL", "llama3:8b")
response = ollama.chat(
    model=model,
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
