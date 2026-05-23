from ollama import chat

response = chat(
    model='qwen3-vl:2b',
    messages=[{'role': 'user', 'content': 'necesito entrenarte para detectar imagenes y reconocer objetos cercanos'}],
)
print(response.message.content)