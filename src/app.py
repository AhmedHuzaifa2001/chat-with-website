import streamlit as st

st.set_page_config(page_title = "chat-with-website" , page_icon = "🤖")

st.title("chat with a website")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Please input a website URL Here!")


user_input = st.chat_input("Type your message here...") 