from langchain_openai import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper 
import os 
import uuid 
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel 

# === 🔐 OpenRouter Setup ===
os.environ["OPENAI_API_KEY"] = os.getenv('ROUTER_API_KEY')
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === 🌤️ Weather API
os.environ["OPENWEATHERMAP_API_KEY"] = "**"
weather = OpenWeatherMapAPIWrapper()

# === 🧠 LLM via OpenRouter (Mixtral)
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0.3
)

# === 📦 State Definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    city: str

# === 🧩 Node 1: Extract city
def agent(state):
    user_input = state["messages"][-1].content

    res = llm.invoke(f"""
    You are given one question and you have to extract the city name from it. 
    Respond ONLY with the city name. If you cannot find a city, respond with an empty string. 

    Here is the question:
    {user_input}
    """)

    city_name = res.content.strip()
    if not city_name:
        return {"messages": [AIMessage(content="I couldn't find a city in your question.")]}
    
    return {
        "messages": [AIMessage(content=f"Extracted city: {city_name}")],
        "city": city_name
    }

# === 🧩 Node 2: Fetch weather
def weather_tool(state):
    city_name = state.get("city", "").strip()

    if not city_name:
        return {"messages": [AIMessage(content="No city name provided. Cannot fetch weather.")]}
    
    weather_info = weather.run(city_name)
    return {"messages": [AIMessage(content=weather_info)]}

# === ⚙️ Workflow Definition
memory = MemorySaver()
workflow = StateGraph(State)

workflow.add_edge(START, "agent")
workflow.add_node("agent", agent)
workflow.add_node("weather", weather_tool)
workflow.add_edge("agent", "weather")
workflow.add_edge("weather", END)

app = workflow.compile(checkpointer=memory)

# === 🧪 Run the graph
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}
user_query = "What is the weather in São Paulo?"

response = app.invoke(
    {"messages": [HumanMessage(content=user_query)]},
    config=config
)

# === 💬 Final Output
print("🌦️ AI:", response["messages"][-1].content)

