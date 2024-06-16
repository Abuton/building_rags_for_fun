import streamlit as st
from client import main
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

rag_history = main()

st.set_page_config(page_title="Complaint Exploration", page_icon="🧑‍💼", layout='wide')

st.title("Complaint Exploration with LLM (llama3) ")
     
st.markdown("# Complaint LLM")
st.sidebar.header("Welcome to the Kitopi Complaint LLM")


# input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# Check if CSV files are uploaded
# if input_csvs:
#     # Select a CSV file from the uploaded files using a dropdown menu
#     selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
#     selected_index = [file.name for file in input_csvs].index(selected_file)

#     #load and display the selected csv file 
#     st.info("CSV uploaded successfully")
#     data = pd.read_csv(input_csvs[selected_index], parse_dates=["Order Date"], skip_blank_lines=True,)
#     st.dataframe(data.head(10), use_container_width=True)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [] # store chat history for context 
    st.session_state.chat_history_messages = [
        {"role": "assistant", "content": "Embeddings has been created based on the CSV selected. Now ask your question Bitch :rabbit: "}
    ] # store chat history for displaying prior messages
    

if prompt := st.chat_input("Your question"): # get user input
    with st.chat_message("user"):
        st.session_state.chat_history_messages.append({"role": "user", "content": prompt}) # storing user input for display


for message in st.session_state.chat_history_messages:
    with st.chat_message(message["role"]):
        st.write(message['content'])


if st.session_state.chat_history_messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_msg = rag_history.invoke({"input": prompt, "chat_history": st.session_state.messages}) # computing response
            if len(ai_msg['answer']) > 0:
                st.session_state.messages.extend([HumanMessage(content=prompt), ai_msg['answer']]) # storing answer for context
                st.write(ai_msg['answer'])
                msg = st.session_state.chat_history_messages.append({"role": "assistant", "content": ai_msg['answer']}) # storing answer for display
                print(ai_msg['chat_history'])
                print(f"Count of Conversation {len(st.session_state.messages)}")

if st.button("Clear Context History"):
    st.session_state.messages.clear()
    st.info("Context Cleared")
