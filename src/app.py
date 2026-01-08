import streamlit as st
from langchain_core.messages import AIMessage , HumanMessage


def get_response(user_input):
    return "I do not know!"

st.set_page_config(page_title = "chat-with-website" , page_icon = "🤖")

st.title("chat with a website")

if "chat_history" not in st.session_state:

    st.session_state.chat_history = [
        AIMessage(content = "Hey I am a bot, How can I help you today?")
    ]

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Please input a website URL Here!")


if website_url is None or website_url == "":
    st.info("Please Enter a url to chat!!!")
else:

    user_input = st.chat_input("Type your message here...") 

    if user_input is not None and user_input != "":
        response = get_response(user_input)
        st.session_state.chat_history.append(HumanMessage(content = user_input))
        st.session_state.chat_history.append(AIMessage(content = response))

    ## Conversation

    for message in st.session_state.chat_history:
        if isinstance(message , AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

        elif isinstance(message , HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)