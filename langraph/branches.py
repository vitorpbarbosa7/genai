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
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
import nest_asyncio

# === 🔐 OpenRouter Setup ===
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === 🤖 LLM
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0.3
)
import subprocess

def render_mermaid_graph(app, mmd_file="graph.mmd", output_image="graph.png"):
    """
    Generate a Mermaid diagram from a LangGraph app and render it as a PNG image.

    Parameters:
    - app: The compiled LangGraph application.
    - mmd_file (str): Filename for the Mermaid definition file.
    - output_image (str): Filename for the output image.
    """
    # Generate Mermaid diagram code
    mermaid_code = app.get_graph().draw_mermaid()

    # Save Mermaid code to .mmd file
    with open(mmd_file, "w") as f:
        f.write(mermaid_code)

    # Use mmdc to convert .mmd to .png
    try:
        subprocess.run(["mmdc", "-i", mmd_file, "-o", output_image], check=True)
        print(f"\n✅ Mermaid diagram saved as {output_image}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to generate diagram: {e}")


# === 📦 State Definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str

# === 🧠 Node 1: Extract topic
def extract_topic(state):
    user_input = state["messages"][-1].content

    res = llm.invoke(f"""
    Extract a one-word topic from the following sentence (like 'cat' or 'dog').
    Respond ONLY with the topic word, nothing else.

    Sentence: {user_input}
    """)
    topic = res.content.strip().lower()

    if not topic:
        return {"messages": [AIMessage(content="I couldn't find a topic.")]}

    return {
        "messages": [AIMessage(content=f"Extracted topic: {topic}")],
        "topic": topic
    }

# === 🐱 Node 2A: Cat Fact
def cat_fact(state):
    try:
        res = requests.get("https://catfact.ninja/fact")
        fact = res.json().get("fact", "Couldn't get a cat fact.")
        return {"messages": [AIMessage(content=f"🐱 Cat fact: {fact}")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Failed to fetch cat fact: {str(e)}")]}

# === 🐶 Node 2B: Dog Image
def dog_image(state):
    try:
        res = requests.get("https://dog.ceo/api/breeds/image/random")
        image_url = res.json().get("message", "No dog image found.")
        return {"messages": [AIMessage(content=f"🐶 Here's a random dog: {image_url}")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Failed to fetch dog image: {str(e)}")]}

# === ❌ Node 2C: Unsupported
def unsupported_topic(state):
    topic = state.get("topic", "unknown")
    return {
        "messages": [AIMessage(content=f"❌ Sorry, I don't support '{topic}' yet. Try asking about cats or dogs.")]
    }

# === 🔀 Router: Choose which node to go to
def router(state):
    topic = state.get("topic", "")
    if "cat" in topic:
        return "cat"
    elif "dog" in topic:
        return "dog"
    else:
        return "unsupported"

# === ⚙️ Build the Graph
memory = MemorySaver()
graph = StateGraph(State)

graph.add_node("extract_topic", extract_topic)
graph.add_node("cat", cat_fact)
graph.add_node("dog", dog_image)
graph.add_node("unsupported", unsupported_topic)

graph.add_conditional_edges(
    "extract_topic",
    router,  # routing function
    {
        "cat": "cat",
        "dog": "dog",
        "unsupported": "unsupported"
    }
)

graph.add_edge("cat", END)
graph.add_edge("dog", END)
graph.add_edge("unsupported", END)
graph.add_edge(START, "extract_topic")

app = graph.compile(checkpointer=memory)
print(app.get_graph().draw_ascii())
render_mermaid_graph(app)



if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    user_query = input("💬 Ask something (e.g. tell me about cats or dogs):\n> ")
    response = app.invoke({"messages": [HumanMessage(content=user_query)]}, config=config)

    print("\n🧠 Final Response:")
    for msg in response["messages"]:
        print(f"{msg.type.capitalize()}: {msg.content}")
