from langchain_openai import ChatOpenAI
import os, requests

# 🔑 OpenRouter
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 🎯 LLM
llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0)

# 🌍 APIs
def get_cat_fact():
    return requests.get("https://catfact.ninja/fact").json()['fact']

def get_advice():
    return requests.get("https://api.adviceslip.com/advice").json()['slip']['advice']

cat = get_cat_fact()
advice = get_advice()

# 🧠 Build the prompt
prompt = f"""
Give me a motivational tweet based on this:

- Cat fact: {cat}
- Advice: {advice}
"""

# 🚀 Run LLM
response = llm.invoke(prompt)
print(response.content)

