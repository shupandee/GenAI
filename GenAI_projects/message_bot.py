import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingface_api_key=HF_API_KEY,
    temperature=0.7,
    max_new_tokens=512
)

# Wrap with ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title("Hugging Face Chatbot with Memory (Structured Messages)")

# Initialize chat history with a system message
if "messages" not in st.session_state:
    st.session_state["messages"] = [SystemMessage(content="You are a helpful assistant.")]

# Display previous chat messages
for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Input box
if prompt := st.chat_input("Type your message here..."):

    # Add HumanMessage
    st.session_state["messages"].append(HumanMessage(content=prompt))

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response with full conversation (including SystemMessage)
    with st.chat_message("assistant"):
        response = model.invoke(st.session_state["messages"])
        st.markdown(response.content)

    # Add AIMessage to history
    st.session_state["messages"].append(AIMessage(content=response.content))
