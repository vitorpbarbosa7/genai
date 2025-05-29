from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os

# Setup OpenRouter
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.3)

# Agent A - Curious
agent_a_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Agent A, a curious researcher always asking questions."),
    ("human", "{input}")
])
agent_a = agent_a_prompt | llm

# Agent B - Skeptic
agent_b_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Agent B, a skeptic who always challenges ideas logically."),
    ("human", "{input}")
])
agent_b = agent_b_prompt | llm

# First question from Agent A
agent_a_output = agent_a.invoke({"input": "What makes quantum entanglement so mysterious?"})
print("\n💬 Agent A says:", agent_a_output.content)

# Pass Agent A's message to Agent B
agent_b_output = agent_b.invoke({"input": agent_a_output.content})
print("\n🤨 Agent B replies:", agent_b_output.content)

