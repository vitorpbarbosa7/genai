#!/usr/bin/env python
# coding: utf-8

# In[50]:


from typing import Annotated, TypedDict, List
import os
import requests
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import gradio as gr
from dotenv import load_dotenv


# In[52]:


# Load environment variables
load_dotenv(override=True)

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

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
    system_prompt = (
        "You are a helpful assistant. If the user wants you to send a push notification, \
        extract the exact message they want to send and reply ONLY with that message.\n"
        "Otherwise, reply normally."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ]
    response = llm.invoke(messages)
    state["messages"].append(response)

    # modify states inside the node
    if "push" in state["messages"][-2]["content"].lower():
        # manually configured the state as push
        state["intent"] = "push"
        # manually configured the push_text as this
        state["push_text"] = response.content.strip()
    else:
        state["intent"] = "end"
    return state

def push_node(state: State) -> State:
    result = push_tool(state["push_text"])
    state["messages"].append({"role": "tool", "name": "send_push_notification", "content": result})
    return state

# === Router ===
def router(state: State) -> str:
    return state["intent"]


# In[53]:


# === Graph ===
graph_builder = StateGraph(State)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("push", push_node)
graph_builder.add_node("router", router)

graph_builder.set_entry_point("llm")
graph_builder.add_edge("llm", "router")
graph_builder.add_conditional_edges("router", router, {
    "push": "push",
    "end": END
})
graph_builder.add_edge("push", END)


# In[55]:


graph = graph_builder.compile()


# In[56]:


display(Image(graph.get_graph().draw_mermaid_png()))


# In[59]:


import traceback

def chat(user_input: str, history):
    try:
        messages = [{"role": "user", "content": user_input}]
        result = graph.invoke({"messages": messages, "intent": "", "push_text": ""})
        return result["messages"][-1]["content"]
    except Exception as e:
        traceback.print_exc()  # full traceback in terminal
        return f"❌ Error: {str(e)}"


gr.ChatInterface(chat, type="messages").launch()


# In[ ]:




