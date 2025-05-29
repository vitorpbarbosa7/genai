from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
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

# === Memory store ===
chat_histories = {}

def memory(session_id: str):
    """Returns chat history object per session ID."""
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

# === Wrap with memory ===
agent_a_with_memory = RunnableWithMessageHistory(
    agent_a,
    lambda session_id: memory(session_id),
    input_messages_key="input",
    history_messages_key="history"
)

agent_b_with_memory = RunnableWithMessageHistory(
    agent_b,
    lambda session_id: memory(session_id + "_b"),
    input_messages_key="input",
    history_messages_key="history"
)

# === Message passing loop ===
def agent_conversation():
    message = "Hi, how are you?"
    session_id = "chat_session_1"

    for round in range(2):
        print(f"\n🔁 Round {round + 1}")

        response_a = agent_a_with_memory.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"\n💬 Agent A: {response_a.content}")
        message = response_a.content

        response_b = agent_b_with_memory.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"\n🤨 Agent B: {response_b.content}")
        message = response_b.content

# === Run the conversation ===
agent_conversation()

