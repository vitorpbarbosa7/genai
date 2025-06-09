import os
import uuid
import subprocess
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# === 🔐 Configuração da API
os.environ["OPENAI_API_KEY"] = os.getenv("ROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === 🤖 Configuração do LLM
llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.3)

# === 📦 Estrutura do Estado
class EstadoDivida(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    entrada_confirmada: bool
    capacidade_pagamento: bool
    data_desejada: bool
    parcelas_definidas: bool
    resolvido: bool

# === 🧠 Chamada de API Simulada
def consultar_divida(state: EstadoDivida) -> EstadoDivida:
    mensagem = AIMessage(content="✅ Simulação: Consulta de dívida realizada.")
    return {"messages": state["messages"] + [mensagem]}

# === 💬 Verificar entrada
def verificar_entrada(state: EstadoDivida) -> EstadoDivida:
    mensagem = AIMessage(content="Você informou valor de entrada? (simulado como sim)")
    return {"messages": state["messages"] + [mensagem], "entrada_confirmada": True}

# === 💬 Verificar capacidade de pagamento
def verificar_capacidade(state: EstadoDivida) -> EstadoDivida:
    mensagem = AIMessage(content="Você informou o valor da parcela que cabe no bolso? (simulado como sim)")
    return {"messages": state["messages"] + [mensagem], "capacidade_pagamento": True}

# === 💬 Verificar data desejada
def verificar_data(state: EstadoDivida) -> EstadoDivida:
    mensagem = AIMessage(content="Você informou a data que deseja pagar? (simulado como sim)")
    return {"messages": state["messages"] + [mensagem], "data_desejada": True}

# === 💬 Verificar número de parcelas
def verificar_parcelas(state: EstadoDivida) -> EstadoDivida:
    mensagem = AIMessage(content="Você informou o número de parcelas desejadas? (simulado como sim)")
    return {"messages": state["messages"] + [mensagem], "parcelas_definidas": True}

# === ✅ Encerrar negociação
def finalizar_negociacao(state: EstadoDivida) -> EstadoDivida:
    mensagem = AIMessage(content="✅ Negociação concluída com base nas informações fornecidas.")
    return {"messages": state["messages"] + [mensagem], "resolvido": True}

# === 🖼️ Gerar imagem do grafo Mermaid
def gerar_imagem_mermaid(app, arquivo_mmd="grafo_divida.mmd", imagem_saida="grafo_divida.png"):
    mermaid = app.get_graph().draw_mermaid()
    with open(arquivo_mmd, "w") as f:
        f.write(mermaid)
    try:
        subprocess.run(["mmdc", "-i", arquivo_mmd, "-o", imagem_saida], check=True)
        print(f"✅ Diagrama Mermaid salvo como {imagem_saida}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao gerar o diagrama: {e}")

# === 🧭 Decidir próximo passo
def decidir_proximo(state: EstadoDivida) -> str:
    if not state.get("entrada_confirmada"):
        return "verificar_entrada"
    elif not state.get("capacidade_pagamento"):
        return "verificar_capacidade"
    elif not state.get("data_desejada"):
        return "verificar_data"
    elif not state.get("parcelas_definidas"):
        return "verificar_parcelas"
    else:
        return "finalizar_negociacao"

# === 🔧 Construir o grafo
memoria = MemorySaver()
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

app = grafo.compile(checkpointer=memoria)
gerar_imagem_mermaid(app)

# === Execução
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    pergunta = input("🧾 Digite sua dúvida sobre dívida:\n> ")
    resposta = app.invoke({"messages": [HumanMessage(content=pergunta)]}, config=config)

    print("\n🧠 Resposta Final:")
    for msg in resposta["messages"]:
        print(f"{msg.type.capitalize()}: {msg.content}")

    print("\n📦 Estado Final:")
    for chave, valor in resposta.items():
        print(f"{chave}: {valor}")

