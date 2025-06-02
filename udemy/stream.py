import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# === 🔐 Setup OpenRouter ===
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === 🤖 LLM with streaming enabled
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0.3,
    streaming=True
)

# === 📦 Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sarcastic assistant who never answers directly."),
    ("human", "{input}")
])

# === 🔧 Lambda to tweak input
prefixer = RunnableLambda(lambda x: {
    "input": f"{x['input']} (and sound annoyed while answering)"
})

# === 🔗 Pipeline (no parser here, because we're streaming)
chain = prefixer | prompt | llm

# === 🧑‍💻 Run with streaming output like typing
def run_streaming_chain(user_input: str):
    print("🤖", end=" ", flush=True)
    for chunk in chain.stream({"input": user_input}):
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
            time.sleep(0.05)  # just for drama
    print("\n")

# === 🚀 Run the chatbot
if __name__ == "__main__":
    run_streaming_chain("Why do people like coffee?")

