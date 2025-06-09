# frontend.py
import streamlit as st
import requests
import uuid

st.set_page_config(page_title="🤝 Assistente de Dívidas", layout="centered")
st.title("💸 Negociação Interativa de Dívidas")

# cria thread de conversa única por aba
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

# caixa de entrada do usuário
user_msg = st.text_input("Sua mensagem:")

# botão de envio
if st.button("Enviar") and user_msg:
    st.session_state.history.append(("Você", user_msg))

    try:
        r = requests.post(
            "http://localhost:8000/ask",
            json={"question": user_msg, "thread_id": st.session_state.thread_id},
            timeout=60,
        )
        r.raise_for_status()           # lança exceção se status != 200
        data = r.json()
    except Exception as err:
        st.error(f"Erro ao chamar backend: {err}")
        st.stop()

    for msg in data["messages"][-2:]:
        st.session_state.history.append(("Assistente", msg))

    if data.get("done"):
        st.session_state.history.append(("Assistente", "✅ Negociação finalizada. Obrigado!"))


# renderiza chat
for speaker, text in st.session_state.history:
    st.write(f"**{speaker}:** {text}")

