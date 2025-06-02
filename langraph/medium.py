from lanchain_openai import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper 
import os 
import uuid 
from langchain.tools import Tool
from langchain.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel 

os.environ["OPENWEATHERMAP_API_KEY"] = "**"
weather = OpenWeatherMapAPIWrapper()

MODEL = "llama3.2"
llm = ChatOllama(model=MODEL)

#** Define the State**
class State(TypedDict):
    messages: Annotated[list, add_messages]
    city: str # Adding city key to track extratec city name

# ** Node 1: Extract city from user input**
def agent(state):
    # Extract latest user message
    user_input = state["messages"][-1].content

    res = llm.invoke(f"""
    You are given one question and you have to extract the city name from it. 
    Responde ONLY with the city name. If you cannot find a city, respond with an empty string. 

    Here is the question:
    {user_input}
    """)

    city_name = res.content.strip()
    if not city_name:
        return {"messages": [AIMessage(content="I couldn't find a city in your question.")]}
    
    # from the llm we extract the city and return here using two keys:
    # messages and city 
    return {"messages": [AIMessage(content=f"Extracted city: {city_name}")], "city":city_name}

# **Node 2: Fetch ewather information**
def weather_tool(state):
    # the weather tool uses state
    # what part of the state?
    # uses the city part
    city_name = state.get("city", "").strip()

    if not city_name:
        return {"messages": [AIMessage(content="No city name provided. Cannot fetch weather.")]}

    # call tool
    weatehr_info = weather.info(city_name)
    return {"messages": [AIMessage(content=weather_info)]}



# Now some workflow 
memory = MemorySaver()
workflow = StateGraph(State)


# ** Define transitions between nodes
# start by some edge called agent?
workflow.add_edge(START, "agent")

# Define the nodes
# basically the agent to ask for the city 
# and the tool to return weather response
workflow.add_node("agent", agent)
workflow.add_node("weather", weather_tool)

# ** Connect Nodes using edges, offcourse **
# after the agent node runs and returns output, redirects to the weather 
workflow.add_edge("agent", "weather")
# after weather runs, it goes to the END of our guy langgraph
workflow.add_edge("weather", END)

# Compile the workflow with memory checkpointer
# checkpointer allows to save the state
app = workflow.compile(checkpointer=memory)


# Create a unique config dictionary to satisfy the checkpointer requirements
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# ** Run the workflow
user_query = "What is the weather in sao paulo?"
# this key corresponds the one in which the first agent will read from 
# here configured as a HumanMessage
response = app.invoke({"messages": [HumanMessage(content=user_query)]}, config = config)

# response will be after running the whole graph 
# from the last message we will know from the last node what is the output after passing throught all nodes
print("AI:", response["messages"][-1].content)






