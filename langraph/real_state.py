import os
import uuid
import subprocess
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# === Configuração da API ===
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.3)

# Encapsular LLM com histórico de mensagens
chat_histories = {}

def memory(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

llm_agent = RunnableWithMessageHistory(
    llm,
    lambda session_id: memory(session_id),
    input_messages_key="input",
    history_messages_key="history"
)

# === Estado ===
class EstadoDivida(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    entrada_confirmada: bool
    valor_entrada: float
    capacidade_pagamento: bool
    valor_parcela: float
    data_desejada: bool
    data_vencimento: str
    parcelas_definidas: bool
    num_parcelas: int
    resolvido: bool

# === Funções dos nós ===
def consultar_divida(state: EstadoDivida) -> EstadoDivida:
    return {
        "messages": state["messages"] + [AIMessage(content="✅ Consulta realizada: Valor devido R$1800.")]
    }

def verificar_entrada(state: EstadoDivida) -> EstadoDivida:
    user_input = state["messages"][-1].content
    prompt = f"O cliente disse: \"{user_input}\". Qual o valor de entrada em reais que ele mencionou? Apenas o número. Se não disse, responda 'nenhum'."
    ai_response = llm.invoke([HumanMessage(content=prompt)])
    content = ai_response.content.strip()
    if content.lower() == "nenhum":
        return {"messages": state["messages"] + [ai_response], "entrada_confirmada": False}
    try:
        valor = float(content.replace("R$", "").replace(",", "."))
        return {
            "messages": state["messages"] + [ai_response],
            "entrada_confirmada": True,
            "valor_entrada": valor
        }
    except:
        return {"messages": state["messages"] + [AIMessage(content="❌ Não entendi o valor da entrada. Pode repetir?")]}

def verificar_capacidade(state: EstadoDivida) -> EstadoDivida:
    user_input = state["messages"][-1].content
    prompt = f"O cliente disse: \"{user_input}\". Qual o valor da parcela que ele pode pagar? Apenas o número. Se não disse, responda 'nenhum'."
    ai_response = llm.invoke([HumanMessage(content=prompt)])
    content = ai_response.content.strip()
    if content.lower() == "nenhum":
        return {"messages": state["messages"] + [ai_response], "capacidade_pagamento": False}
    try:
        valor = float(content.replace("R$", "").replace(",", "."))
        return {
            "messages": state["messages"] + [ai_response],
            "capacidade_pagamento": True,
            "valor_parcela": valor
        }
    except:
        return {"messages": state["messages"] + [AIMessage(content="❌ Não entendi o valor da parcela. Pode repetir?")]}

def verificar_data(state: EstadoDivida) -> EstadoDivida:
    user_input = state["messages"][-1].content
    prompt = f"O cliente disse: \"{user_input}\". Qual a data desejada de vencimento? Formato DD/MM/AAAA. Se não disse, responda 'nenhuma'."
    ai_response = llm.invoke([HumanMessage(content=prompt)])
    content = ai_response.content.strip()
    if content.lower() == "nenhuma":
        return {"messages": state["messages"] + [ai_response], "data_desejada": False}
    return {
        "messages": state["messages"] + [ai_response],
        "data_desejada": True,
        "data_vencimento": content
    }

def verificar_parcelas(state: EstadoDivida) -> EstadoDivida:
    user_input = state["messages"][-1].content
    prompt = f"O cliente disse: \"{user_input}\". Em quantas parcelas ele deseja pagar? Apenas número inteiro. Se não disse, responda 'nenhum'."
    ai_response = llm.invoke([HumanMessage(content=prompt)])
    content = ai_response.content.strip()
    if content.lower() == "nenhum":
        return {"messages": state["messages"] + [ai_response], "parcelas_definidas": False}
    try:
        n = int(content)
        return {
            "messages": state["messages"] + [ai_response],
            "parcelas_definidas": True,
            "num_parcelas": n
        }
    except:
        return {"messages": state["messages"] + [AIMessage(content="❌ Não entendi o número de parcelas. Pode repetir?")]}

def finalizar_negociacao(state: EstadoDivida) -> EstadoDivida:
    total_pago = state.get("valor_entrada", 0) + state.get("valor_parcela", 0) * state.get("num_parcelas", 0)
    return {
        "messages": state["messages"] + [
            AIMessage(content=f"📄 Negociação finalizada: Entrada de R${state.get('valor_entrada', 0)}, {state.get('num_parcelas', 0)}x R${state.get('valor_parcela', 0)} com vencimento em {state.get('data_vencimento', '')}. Total: R${total_pago}.")
        ],
        "resolvido": True
    }

# === Roteador ===
def decidir_proximo(state: EstadoDivida) -> str:
    if not state.get("entrada_confirmada"):
        return "verificar_entrada"
    if not state.get("capacidade_pagamento"):
        return "verificar_capacidade"
    if not state.get("data_desejada"):
        return "verificar_data"
    if not state.get("parcelas_definidas"):
        return "verificar_parcelas"
    return "finalizar_negociacao"

# === Construir grafo ===
memory = MemorySaver()
grafo = StateGraph(EstadoDivida)

grafo.add_node("consultar_divida", consultar_divida)
grafo.add_node("verificar_entrada", verificar_entrada)
grafo.add_node("verificar_capacidade", verificar_capacidade)
grafo.add_node("verificar_data", verificar_data)
grafo.add_node("verificar_parcelas", verificar_parcelas)
grafo.add_node("finalizar_negociacao", finalizar_negociacao)

grafo.add_edge(START, "consultar_divida")
for origem in ["consultar_divida", "verificar_entrada", "verificar_capacidade", "verificar_data", "verificar_parcelas"]:
    grafo.add_conditional_edges(origem, decidir_proximo, {
        "verificar_entrada": "verificar_entrada",
        "verificar_capacidade": "verificar_capacidade",
        "verificar_data": "verificar_data",
        "verificar_parcelas": "verificar_parcelas",
        "finalizar_negociacao": "finalizar_negociacao"
    })

grafo.add_edge("finalizar_negociacao", END)

app = grafo.compile(checkpointer=memory)

# === Execução ===
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        user_input = input("💬 Cliente: ")
        resposta = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        print("\n🧠 Resposta:")
        for msg in resposta["messages"][-2:]:
            print(f"{msg.type.capitalize()}: {msg.content}")

        if resposta.get("resolvido"):
            print("\n✅ Negociação finalizada.")
            break

