
import os
import requests

# 🔑 Your Hugging Face API key
api_key = os.getenv("HF_TOKEN")

# 🔥 Choose the model:
# Examples:
# - meta-llama/Meta-Llama-3-8B-Instruct
# - microsoft/Phi-3-mini-4k-instruct
# - mistralai/Mistral-7B-Instruct-v0.2

model = "microsoft/Phi-3-mini-4k-instruct"

# Hugging Face inference endpoint
url = f"https://api-inference.huggingface.co/models/{model}"

# Headers with the token
headers = {
            "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
                }

# Payload with your input prompt
data = {
            "inputs": "Explain quantum computing in simple terms.",
                "parameters": {
                            "max_new_tokens": 500,
                                    "temperature": 0.2
                                        }
                }

# POST request to the API
response = requests.post(url, headers=headers, json=data)

# Convert to JSON
result = response.json()

print(result)

