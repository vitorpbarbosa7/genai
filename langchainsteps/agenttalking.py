from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

# === Setup OpenRouter ===
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === Shared LLM ===
llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.3)

# === Directive ===
BASE_DIRECTIVE = "Respond in no more than two sentences. Keep it concise and relevant."

# === PromptBuilder Class ===
class PromptBuilder:
    def __init__(self, directive: str):
        self.directive = directive

    def create_prompt(self, role_description: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", f"{role_description} {self.directive}"),
            ("human", "{input}")
        ])

# === Instantiate the builder with the directive ===
builder = PromptBuilder(BASE_DIRECTIVE)

# === Create agent prompts ===
agent_a_prompt = builder.create_prompt(
    "You are a teenager guy in high school, extroverted, into anti-system music."
)
agent_b_prompt = builder.create_prompt(
    "You are a teenager girl in high school, introverted, into psychology books."
)

# === Bind agents ===
agent_a = agent_a_prompt | llm
agent_b = agent_b_prompt | llm

# === Message passing loop ===
def agent_conversation():
    message = "Hi, how are you?"

    for round in range(2):
        print(f"\n🔁 Round {round + 1}")

        response_a = agent_a.invoke({"input": message})
        print(f"\n💬 Agent A: {response_a.content}")
        message = response_a.content

        response_b = agent_b.invoke({"input": message})
        print(f"\n🤨 Agent B: {response_b.content}")
        message = response_b.content

# === Run ===
agent_conversation()

