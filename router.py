from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 🔑 API setup
import os
os.environ["OPENAI_API_KEY"] = os.getenv('ROUTER_API_KEY')
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 🎯 Initialize the LLM
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",  # Or other models available on OpenRouter
    temperature=0.7
)

# 🧠 Add memory to keep conversation context
memory = ConversationBufferMemory()

# 🔗 Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # Prints the steps
)

# 💬 Interact
result = conversation.run("Hi, who won the world cup in 2018?")
print(result)

result = conversation.run("And who was the best player?")
print(result)

