from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os

# Configurar OpenRouter (supondo variáveis de ambiente definidas corretamente)
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Inicializar modelo via OpenRouter
llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.3)

# Criar mensagens
messages = [
    SystemMessage(content="Você é um assistente e está respondendo perguntas gerais."),
    HumanMessage(content="Explique para mim brevemente o conceito de IA.")
]

# Executar a chamada
response = llm.invoke(messages)

# Mostrar resultado
print(response.content)
