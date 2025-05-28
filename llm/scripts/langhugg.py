import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

# Load the Hugging Face token from .env
load_dotenv()

# Get your Hugging Face API token
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_key:
    raise Exception("Missing Hugging Face API token in environment variable HUGGINGFACEHUB_API_TOKEN.")

# Initialize the LLM with explicit parameters
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    huggingfacehub_api_token=api_key,
    temperature=0.2,
    max_new_tokens=500
)

# Run a simple prompt
prompt = "Explain quantum computing in simple terms."

result = llm.invoke(prompt)

print(result)

