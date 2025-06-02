from lanchain_openai import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper 
import os 
import uuid 
from langchain.tools import Tool
from langchain.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumamMessage
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

