from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

# 🔑 API setup
os.environ["OPENAI_API_KEY"] = os.getenv('ROUTER_API_KEY')
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 🎯 Initialize the LLM
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",  # Change to other models if desired
    temperature=0.7
)

# 🧠 Add memory to keep conversation context
memory = ConversationBufferMemory()

# 🔗 Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 📄 Markdown output file
output_file = "output.md"

def save_to_markdown(prompt, response):
    with open(output_file, "a") as f:
        f.write(f"## User:\n{prompt}\n\n")
        f.write(f"## Assistant:\n{response}\n\n---\n\n")

# 💬 Interact
q1 = "Hi, who won the world cup in 2018?"
a1 = conversation.run(q1)
print(a1)
save_to_markdown(q1, a1)

q2 = "And who was the best player?"
a2 = conversation.run(q2)
print(a2)
save_to_markdown(q2, a2)

