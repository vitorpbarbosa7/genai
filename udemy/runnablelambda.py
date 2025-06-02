# === 🔧 Setup básico ===
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# === 🔐 OpenRouter Config ===
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === 🤖 LLM via OpenRouter ===
llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.3)

# === 🧠 Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sarcastic assistant who never answers directly."),
    ("human", "{input}")
])

# === 🧱 Lambda para modificar o input antes do prompt
prefixer = RunnableLambda(lambda x: {
    "input": f"{x['input']} (and be sarcastic about it!)"
})

# === 🧰 Parser para extrair string pura
parser = StrOutputParser()

# === 🔗 Encadeamento com pipeline funcional
chain = prefixer | prompt | llm | parser

# === 🚀 Rodar o agente
if __name__ == "__main__":
    user_input = "How do airplanes fly?"
    result = chain.invoke({"input": user_input})
    print("\n🛩️ Final Response:\n", result)

