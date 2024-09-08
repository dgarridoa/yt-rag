import requests
import streamlit as st

from yt_rag.settings import get_settings

settings = get_settings()
url = "http://localhost:5000/rag"

st.title("💬 Chatbot")
st.caption("🚀 RAG powered by LLama 3.1")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I'm here to help you with any questions you have about ThePrimeagen, a YouTuber who creates content related to software engineering. Feel free to ask me anything, and I'll use transcriptions from his videos to provide you with accurate and helpful answers.",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = requests.post(
        url,
        auth=(
            settings.api_username.get_secret_value(),
            settings.api_password.get_secret_value(),
        ),
        params={"input": st.session_state.messages[-1]["content"]},
        timeout=120,
    ).json()
    msg = response["answer"]
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    video_url = (
        f'https://www.youtube.com/watch?v={response["context"][0]["video_id"]}'
    )
    st.video(video_url)
