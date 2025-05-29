from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os 

# 🔑 OpenRouter setup
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"


# ✅ Initialize the LLM (Chat Model)
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0.2)

# 🔥 Define the chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a wise assistant that answers questions clearly."),
    ("human", "Explain the concept of entropy in one paragraph.")
])

chain = prompt | llm 

result = chain.invoke({"topic": "quantum entanglement"})


