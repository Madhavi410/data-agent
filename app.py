"""
Streamlit web UI for the Data Agent.
Run locally:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from data_agent import DataAgent

st.set_page_config(page_title="Data Agent", page_icon="📊", layout="centered")
st.title("📊 Data Agent")
st.caption("Ask questions about your CSV data in plain English.")


# ---------- config from Streamlit secrets / environment ----------
def secret(name: str):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


api_key = secret("ANTHROPIC_API_KEY")
if not api_key:
    st.error("ANTHROPIC_API_KEY is not set. Add it in Streamlit secrets or as an environment variable.")
    st.stop()
os.environ["ANTHROPIC_API_KEY"] = api_key

# ---------- optional password gate (protects your API credits) ----------
app_password = secret("APP_PASSWORD")
if app_password:
    if st.session_state.get("authed") is not True:
        pw = st.text_input("Password", type="password")
        if pw:
            if pw == app_password:
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
        st.stop()

# ---------- choose data ----------
uploaded = st.file_uploader("Upload a CSV (or use the sample data)", type="csv")

if uploaded is not None:
    data_key = f"{uploaded.name}-{uploaded.size}"
    csv_path = None
else:
    data_key = "sample"
    csv_path = os.path.join(os.path.dirname(__file__), "sales.csv")

# Create a fresh agent whenever the dataset changes
if st.session_state.get("data_key") != data_key:
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue())
            csv_path = tmp.name
    st.session_state.agent = DataAgent(csv_path)
    st.session_state.messages = []
    st.session_state.data_key = data_key

agent = st.session_state.agent

with st.expander(f"Data preview ({len(agent.df)} rows x {len(agent.df.columns)} columns)"):
    st.dataframe(agent.df.head(20), width="stretch")

# ---------- chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            try:
                answer = agent.ask(question, verbose=False)
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
