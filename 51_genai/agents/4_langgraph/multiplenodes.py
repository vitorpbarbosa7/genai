from typing import Annotated, TypedDict, List
import os
import requests
import json
import traceback
import gradio as gr
import subprocess
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# === Load env ===
load_dotenv(override=True)

# === State ===
class State(TypedDict):
    messages: Annotated[List[dict], add_messages]
    intent: str
    num1: float
    num2: float
    result: float

# === LLM ===
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("ROUTER_API_KEY"),
)

# === Nodes ===
def llm_parser_node(state: State) -> State:
    system_prompt = {
        "role": "system",
        "content": (
            "You are a JSON-only assistant. Extract the operation and two numbers from the user request."
            " Respond ONLY with a JSON object like:\n"
            '{ "intent": "sum", "num1": 10, "num2": 20 }\n'
            "Possible intents: 'sum', 'subtract', 'multiply', 'divide'."
        )
    }
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)

    try:
        parsed = json.loads(response.content.strip())
        state["intent"] = parsed.get("intent", "sum")
        state["num1"] = float(parsed.get("num1", 0))
        state["num2"] = float(parsed.get("num2", 0))
    except Exception:
        state["intent"] = "sum"
        state["num1"] = 0
        state["num2"] = 0
        state["messages"].append({"role": "assistant", "content": f"❌ Failed to parse JSON: {response.content}"})
        return state

    state["messages"].append({"role": "assistant", "content": response.content})
    return state

def sum_node(state: State) -> State:
    state["result"] = state["num1"] + state["num2"]
    return state

def subtract_node(state: State) -> State:
    state["result"] = state["num1"] - state["num2"]
    return state

def multiply_node(state: State) -> State:
    state["result"] = state["num1"] * state["num2"]
    return state

def divide_node(state: State) -> State:
    try:
        state["result"] = state["num1"] / state["num2"]
    except ZeroDivisionError:
        state["result"] = float("inf")
    return state

def final_node(state: State) -> State:
    result_msg = f"✅ Resultado: {state['num1']} {state['intent']} {state['num2']} = {state['result']}"
    state["messages"].append({"role": "assistant", "content": result_msg})
    return state

# === Router ===
def math_router(state: State) -> str:
    return {
        "sum": "sum",
        "subtract": "subtract",
        "multiply": "multiply",
        "divide": "divide"
    }.get(state.get("intent", "sum"), "sum")

# === Render Mermaid Diagram ===
def render_mermaid_graph(app, mmd_file="graph.mmd", output_image="graph.png"):
    mermaid_code = app.get_graph().draw_mermaid()
    with open(mmd_file, "w") as f:
        f.write(mermaid_code)
    try:
        subprocess.run(["mmdc", "-i", mmd_file, "-o", output_image], check=True)
        print(f"✅ Diagrama salvo como {output_image}")
    except Exception as e:
        print(f"❌ Falha ao gerar diagrama: {e}")

# === Build Graph ===
graph_builder = StateGraph(State)

graph_builder.add_node("llm_parser", llm_parser_node)
graph_builder.add_node("sum", sum_node)
graph_builder.add_node("subtract", subtract_node)
graph_builder.add_node("multiply", multiply_node)
graph_builder.add_node("divide", divide_node)
graph_builder.add_node("final", final_node)

graph_builder.set_entry_point("llm_parser")
graph_builder.add_conditional_edges("llm_parser", math_router, {
    "sum": "sum",
    "subtract": "subtract",
    "multiply": "multiply",
    "divide": "divide"
})
graph_builder.add_edge("sum", "final")
graph_builder.add_edge("subtract", "final")
graph_builder.add_edge("multiply", "final")
graph_builder.add_edge("divide", "final")
graph_builder.add_edge("final", END)

graph = graph_builder.compile()
render_mermaid_graph(graph)

# === Gradio Chat ===
def chat(user_input: str, history):
    try:
        messages = [{"role": "user", "content": user_input}]
        result = graph.invoke({
            "messages": messages,
            "intent": "",
            "num1": 0,
            "num2": 0,
            "result": 0
        })
        return result["messages"][-1].content
    except Exception as e:
        traceback.print_exc()
        return f"❌ Erro: {str(e)}"

# === Launch UI ===
gr.ChatInterface(
    fn=chat,
    type="messages",
    title="🧠 Calculadora com LLM + LangGraph",
    description="Peça uma operação matemática: somar, subtrair, multiplicar, dividir."
).launch()

