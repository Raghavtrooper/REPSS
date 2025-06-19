import streamlit as st
import time


def clear_chat_callback():
    """
    Clears the chat history in Streamlit's session state.
    """
    st.session_state.messages = []
    st.rerun()
