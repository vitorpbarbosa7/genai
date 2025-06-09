from typing import Annotated, TypedDict, List
import os
import requests
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import gradio as gr
import json
import traceback

# === Load env ===
load_dotenv(override=True)

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

# === Tool ===
def push_tool(msg: str) -> str:
    try:
        requests.post(pushover_url, data={"token": pushover_token, "user": pushover_user, "message": msg})
        return "✅ Push notification sent."
    except Exception as e:
        return f"❌ Failed to send push: {str(e)}"

# === State ===
class State(TypedDict):
    messages: Annotated[List[dict], add_messages]
    intent: str
    push_text: str

# === LLM ===
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("ROUTER_API_KEY"),
)

# === Nodes ===
def llm_node(state: State) -> State:
    system_prompt = {
        "role": "system",
        "content": (
            "You are a JSON-only assistant. Given a user request, respond with a JSON object like:\n\n"
            '{\n  "intent": "push",\n  "push_text": "message to send"\n}\n\n'
            "If no push is needed, reply:\n"
            '{\n  "intent": "end",\n  "push_text": ""\n}\n\n'
            "Respond ONLY with JSON. Do not explain or talk outside the JSON."
        )
    }
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)

    # Try to parse JSON response
    try:
        raw = response.content.strip()
        parsed = json.loads(raw)
        state["intent"] = parsed.get("intent", "end")
        state["push_text"] = parsed.get("push_text", "")
    except Exception as e:
        state["intent"] = "end"
        state["push_text"] = ""
        state["messages"].append({
            "role": "assistant",
            "content": f"❌ Failed to parse JSON. Raw: {response.content}"
        })
        return state

    # Also append the actual assistant message
    state["messages"].append({
        "role": "assistant",
        "content": response.content
    })
    return state

def push_node(state: State) -> State:
    result = push_tool(state["push_text"])
    breakpoint()
    state["messages"].append({"role": "tool", "name": "send_push_notification", "content": result})
    return state

def route_decision(state: State) -> str:
    return state["intent"]

# === Graph ===
graph_builder = StateGraph(State)

graph_builder.add_node("llm", llm_node)
graph_builder.add_node("push", push_node)

# ✅ Only set conditional routing via add_conditional_edges
graph_builder.set_entry_point("llm")
graph_builder.add_conditional_edges("llm", route_decision, {
    "push": "push",
    "end": END
})
graph_builder.add_edge("push", END)

graph = graph_builder.compile()

# === Gradio UI ===
def chat(user_input: str, history):
    try:
        messages = [{"role": "user", "content": user_input}]
        result = graph.invoke({"messages": messages, "intent": "", "push_text": ""})
        return result["messages"][-1]["content"]
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error: {str(e)}"

# Launch Gradio app
gr.ChatInterface(chat, type="messages").launch()

