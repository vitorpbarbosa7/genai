#main logic
import os, uuid
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# 🔑  OpenRouter keys (ajuste conforme seu ambiente)
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 🚀  LLM
llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.3)

# 🗂️  Histórico de conversa por thread
_histories = {}

def _history(thread_id: str):
    if thread_id not in _histories:
        _histories[thread_id] = InMemoryChatMessageHistory()
    return _histories[thread_id]

# Wrapper que mantém histórico automaticamente
llm_runner = RunnableWithMessageHistory(
    llm,
    lambda cfg: _history(cfg["configurable"]["thread_id"]),
    input_messages_key="messages",  # LangGraph já envia lista em "messages"
    history_messages_key="history"
)

# 📦  Estrutura de Estado
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

# === Nós ==============================================================

def _query_llm(prompt: str, state: EstadoDivida):
    """Envia prompt ao LLM mantendo o histórico."""
    cfg = {"configurable": {"thread_id": state.get("thread_id", "tmp")}}
    resp = llm_runner.invoke({"messages": [HumanMessage(content=prompt)]}, config=cfg)
    return resp.content.strip()

# 0) consulta inicial
def consultar_divida(state: EstadoDivida):
    return {"messages": state["messages"] + [AIMessage(content="✅ Consulta realizada: Valor devido R$1800.")]}

# 1) entrada
def verificar_entrada(state: EstadoDivida):
    prompt = (
        f"O cliente disse: '{state['messages'][-1].content}'. "
        "Qual o valor de entrada em reais que ele mencionou? Somente número ou 'nenhum'."
    )
    resp = _query_llm(prompt, state)
    if resp.lower() == "nenhum":
        return {"messages": state["messages"] + [AIMessage(content=resp)]}
    try:
        valor = float(resp.replace("R$", "").replace(",", "."))
        return {"messages": state["messages"] + [AIMessage(content=f"Entrada registrada: R${valor}")],
                "entrada_confirmada": True, "valor_entrada": valor}
    except:
        return {"messages": state["messages"] + [AIMessage(content="❌ Não entendi a entrada. Informe novamente.")]}

# 2) capacidade
def verificar_capacidade(state: EstadoDivida):
    prompt = (
        f"O cliente disse: '{state['messages'][-1].content}'. "
        "Qual o valor da parcela que cabe no bolso? Somente número ou 'nenhum'."
    )
    resp = _query_llm(prompt, state)
    if resp.lower() == "nenhum":
        return {"messages": state["messages"] + [AIMessage(content=resp)]}
    try:
        valor = float(resp.replace("R$", "").replace(",", "."))
        return {"messages": state["messages"] + [AIMessage(content=f"Parcela registrada: R${valor}")],
                "capacidade_pagamento": True, "valor_parcela": valor}
    except:
        return {"messages": state["messages"] + [AIMessage(content="❌ Não entendi a parcela. Informe novamente.")]}

# 3) data
def verificar_data(state: EstadoDivida):
    prompt = (
        f"O cliente disse: '{state['messages'][-1].content}'. "
        "Qual a data de vencimento desejada (DD/MM/AAAA) ou 'nenhuma'?"
    )
    resp = _query_llm(prompt, state)
    if resp.lower() == "nenhuma":
        return {"messages": state["messages"] + [AIMessage(content=resp)]}
    return {"messages": state["messages"] + [AIMessage(content=f"Data registrada: {resp}")],
            "data_desejada": True, "data_vencimento": resp}

# 4) parcelas
def verificar_parcelas(state: EstadoDivida):
    prompt = (
        f"O cliente disse: '{state['messages'][-1].content}'. "
        "Em quantas parcelas deseja pagar? Número inteiro ou 'nenhum'."
    )
    resp = _query_llm(prompt, state)
    if resp.lower() == "nenhum":
        return {"messages": state["messages"] + [AIMessage(content=resp)]}
    if resp.isdigit():
        n = int(resp)
        return {"messages": state["messages"] + [AIMessage(content=f"Parcelas registradas: {n}")],
                "parcelas_definidas": True, "num_parcelas": n}
    return {"messages": state["messages"] + [AIMessage(content="❌ Não entendi o número de parcelas. Informe novamente.")]}

# 5) final
def finalizar_negociacao(state: EstadoDivida):
    total = state.get("valor_entrada",0)+state.get("valor_parcela",0)*state.get("num_parcelas",0)
    msg = (
        f"📄 Negociação finalizada: Entrada R${state.get('valor_entrada',0)}, "
        f"{state.get('num_parcelas',0)}x R${state.get('valor_parcela',0)} vencendo em {state.get('data_vencimento','')}. "
        f"Total: R${total}."
    )
    return {"messages": state["messages"] + [AIMessage(content=msg)], "resolvido": True}

# === Router ===
def router(state: EstadoDivida):
    if not state.get("entrada_confirmada"):
        return "verificar_entrada"
    if not state.get("capacidade_pagamento"):
        return "verificar_capacidade"
    if not state.get("data_desejada"):
        return "verificar_data"
    if not state.get("parcelas_definidas"):
        return "verificar_parcelas"
    return "finalizar_negociacao"

# === Montagem do Grafo ===
mem = MemorySaver()
G = StateGraph(EstadoDivida)
G.add_node("consultar_divida", consultar_divida)
G.add_node("verificar_entrada", verificar_entrada)
G.add_node("verificar_capacidade", verificar_capacidade)
G.add_node("verificar_data", verificar_data)
G.add_node("verificar_parcelas", verificar_parcelas)
G.add_node("finalizar_negociacao", finalizar_negociacao)
G.add_edge(START, "consultar_divida")
for n in ["consultar_divida", "verificar_entrada", "verificar_capacidade", "verificar_data", "verificar_parcelas"]:
    G.add_conditional_edges(n, router, {
        "verificar_entrada": "verificar_entrada",
        "verificar_capacidade": "verificar_capacidade",
        "verificar_data": "verificar_data",
        "verificar_parcelas": "verificar_parcelas",
        "finalizar_negociacao": "finalizar_negociacao"
    })
G.add_edge("finalizar_negociacao", END)

app = G.compile(checkpointer=mem)
