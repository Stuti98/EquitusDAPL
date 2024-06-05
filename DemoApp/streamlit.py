# app.py

import streamlit as st
from run import query_generator

# Initialize the chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.title("SPARQL Query Generator")

# Form for user input
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You: ", key="input")
    submit_button = st.form_submit_button(label='Send')

# When the user submits a message
if submit_button and user_input:
    # Get the response from the function
    response = query_generator(user_input)
    # Append the user input and response to the chat history
    st.session_state['chat_history'].append({"user": user_input, "response": response})
    # Display the chat history
    # for chat in st.session_state['chat_history']:
    #     st.write(f"You: {chat['user']}", unsafe_allow_html=True)
    #     st.write(f"<span style='color: blue;'>Response:</span> {chat['response']}", unsafe_allow_html=True)

    for chat in st.session_state['chat_history']:
        st.markdown(f"""
        <div style='display: flex; align-items: flex-start;'>
            <div style='margin-right: 10px; font-weight: bold;'>You:</div>
            <div>{chat['user']}</div>
        </div>
        <div style='display: flex; align-items: flex-start;'>
            <div style='margin-right: 10px; font-weight: bold; color: grey;'>Response:</div>
            <div>{chat['response']}</div>
        </div>
        """, unsafe_allow_html=True)
