# === 🧠 LangChain Agent Pipeline with Runnable Composition ===

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# === 🌐 Setup OpenRouter ===
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === 🤖 Shared LLM via OpenRouter ===
llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.3)

# === 🧱 Output parser to get plain string from model
parser = StrOutputParser()

# === 🧰 PromptBuilder Class ===
class PromptBuilder:
    def __init__(self, directive: str):
        self.directive = directive

    def create_prompt(self, role_description: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", f"{role_description} {self.directive}"),
            ("human", "{input}")
        ])

# === 🧠 Instantiate PromptBuilder with directive
BASE_DIRECTIVE = "Respond in no more than two sentences. Keep it concise and relevant."
builder = PromptBuilder(BASE_DIRECTIVE)

# === 🎭 Create agent prompts
agent_a_prompt = builder.create_prompt(
    "You are a teenager guy in high school, extroverted, into anti-system music."
)
agent_b_prompt = builder.create_prompt(
    "You are a teenager girl in high school, introverted, into psychology books."
)

# === 🪄 Compose agents using pipeline chaining
agent_a_chain = agent_a_prompt | llm | parser
agent_b_chain = agent_b_prompt | llm | parser

# === 🔁 Simple conversation loop
def agent_conversation():
    message = "Hi, how are you?"

    for round in range(2):
        print(f"\n🔁 Round {round + 1}")

        response_a = agent_a_chain.invoke({"input": message})
        print(f"\n💬 Agent A: {response_a}")
        message = response_a

        response_b = agent_b_chain.invoke({"input": message})
        print(f"\n🤨 Agent B: {response_b}")
        message = response_b

# === 🚀 Run the conversation
if __name__ == "__main__":
    agent_conversation()

