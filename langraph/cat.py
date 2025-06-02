import os
import uuid
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

# === 🔐 OpenRouter Setup ===
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === 💬 LLM via OpenRouter
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0.3
)

# === 📦 State Definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str

# === 🧩 Node 1: Extract topic
def extract_topic(state):
    user_input = state["messages"][-1].content

    res = llm.invoke(f"""
    You are given a message. Extract a one-word topic from it. For example:
    - "Tell me about dogs" → "dog"
    - "What are some cat facts?" → "cat"
    - "What do you know about space?" → "space"

    Respond with ONLY the topic word, nothing else.

    Message: {user_input}
    """)
    topic = res.content.strip().lower()

    if not topic:
        return {"messages": [AIMessage(content="I couldn't understand a topic.")]}

    return {
        "messages": [AIMessage(content=f"Extracted topic: {topic}")],
        "topic": topic
    }

# === 🧩 Node 2: Fetch info from free API
def fetch_fact(state):
    topic = state.get("topic", "").strip().lower()

    if topic != "cat":
        return {"messages": [AIMessage(content=f"I only support cat facts right now. Topic was: {topic}")]}

    try:
        response = requests.get("https://catfact.ninja/fact")
        data = response.json()
        fact = data.get("fact", "No cat fact available.")
        return {"messages": [AIMessage(content=f"🐱 Did you know? {fact}")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Failed to fetch fact: {str(e)}")]}

# === ⚙️ LangGraph Wiring
memory = MemorySaver()
workflow = StateGraph(State)

workflow.add_node("extract_topic", extract_topic)
workflow.add_node("fetch_fact", fetch_fact)
workflow.add_edge(START, "extract_topic")
workflow.add_edge("extract_topic", "fetch_fact")
workflow.add_edge("fetch_fact", END)

app = workflow.compile(checkpointer=memory)

# === 🚀 Run the graph
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

user_query = "Can you tell me something interesting about spac?"
response = app.invoke(
    {"messages": [HumanMessage(content=user_query)]},
    config=config
)

# === 📦 Final Output
print("🧠 Final Response:")
for msg in response["messages"]:
    print(f"{msg.type.capitalize()}: {msg.content}")

