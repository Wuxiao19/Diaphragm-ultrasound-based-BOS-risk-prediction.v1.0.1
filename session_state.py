import streamlit as st
from random import randint
from session_state import get_session_state

state = get_session_state()
if not state.widget_key:
    state.widget_key = str(randint(1000, 100000000))
uploaded_file = st.file_uploader(
    "Choose a file", accept_multiple_files=True, key=state.widget_key)
if st.button('clear uploaded_file'):
    state.widget_key = str(randint(1000, 100000000))
state.sync()

