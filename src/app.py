import streamlit as st
from langchain_core.messages import AIMessage , HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()
import os



## url loading
def document_loading(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    ## Splitting into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    ## Vector stores
    vector_store = Chroma.from_documents(documents = document_chunks , embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    return vector_store

def get_context_retriever_chain(vector_store):


    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(api_key = api_key , model = "llama-3.3-70b-versatile")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name = "chat_history"),
            ("user" , "{input}"),
            ## the below prompt will improvise the user query to a better version
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")

        ]
    )

    history_aware_chain = create_history_aware_retriever(llm = llm , retriever = retriever , prompt = prompt)
    return history_aware_chain



def get_conversational_rag_chain(retriever_chain):

    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(api_key = api_key , model = "llama-3.3-70b-versatile")

    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),

      MessagesPlaceholder(variable_name="chat_history"),

      ("user", "{input}"),
    ])

    stuff_documents = create_stuff_documents_chain(llm = llm , prompt = prompt)

    return create_retrieval_chain(retriever_chain , stuff_documents)



def get_response(user_input):


    history_retriever = get_context_retriever_chain(st.session_state.vectors)
    conversational_rag_chain = get_conversational_rag_chain(history_retriever)

    response = conversational_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']


st.set_page_config(page_title = "Chat with Website 🤖" , page_icon = "🤖", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #8B5CF6, #3B82F6, #EC4899, #06B6D4);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main Container Glassmorphism */
    .main .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        max-width: 1200px;
        margin: 2rem auto;
    }
    
    /* Title Styling */
    h1 {
        background: linear-gradient(135deg, #fff 0%, #E0E7FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 40px rgba(255, 255, 255, 0.3);
        animation: titleGlow 3s ease-in-out infinite;
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.5)); }
        50% { filter: drop-shadow(0 0 30px rgba(236, 72, 153, 0.7)); }
    }
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* Sidebar Header */
    [data-testid="stSidebar"] h2 {
        color: white;
        font-weight: 600;
        font-size: 1.5rem;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1.5rem;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.6);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    
    .stTextInput > label {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    
    /* Chat Input */
    .stChatInput > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stChatInput input {
        color: white;
        font-size: 1rem;
    }
    
    .stChatInput input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Chat Messages - AI */
    .stChatMessage[data-testid="chat-message-AI"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 18px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.2);
        animation: slideInLeft 0.5s ease;
        transition: all 0.3s ease;
    }
    
    .stChatMessage[data-testid="chat-message-AI"]:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 24px rgba(139, 92, 246, 0.3);
    }
    
    /* Chat Messages - Human */
    .stChatMessage[data-testid="chat-message-Human"] {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(236, 72, 153, 0.3);
        border-radius: 18px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 16px rgba(236, 72, 153, 0.2);
        animation: slideInRight 0.5s ease;
        transition: all 0.3s ease;
    }
    
    .stChatMessage[data-testid="chat-message-Human"]:hover {
        transform: translateX(-5px);
        box-shadow: 0 6px 24px rgba(236, 72, 153, 0.3);
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Chat Message Text */
    .stChatMessage p {
        color: white;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Info Box */
    .stAlert {
        background: rgba(59, 130, 246, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        color: white;
        padding: 1.5rem;
        animation: fadeIn 0.6s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #8B5CF6 0%, #EC4899 100%);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #A78BFA 0%, #F472B6 100%);
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #8B5CF6 !important;
        border-right-color: #EC4899 !important;
    }
    
    /* Buttons (if any) */
    .stButton > button {
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(139, 92, 246, 0.5);
        background: linear-gradient(135deg, #A78BFA 0%, #F472B6 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>💬 Chat with Any Website</h1>", unsafe_allow_html=True)



with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Please input a website URL Here!")


if website_url is None or website_url == "":
    st.info("Please Enter a url to chat!!!")

else:

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = [
            AIMessage(content = "Hey I am a bot, How can I help you today?")
        ]

    if "vectors" not in st.session_state:
        st.session_state.vectors = document_loading(website_url)


   

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