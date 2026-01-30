import streamlit as st
from random import randint
from .session_state import get_session_state
from streamlit import session_state as _state

_PERSIST_STATE_KEY = f"{__name__}_PERSIST"



state = get_session_state()
if not state.widget_key:
    state.widget_key = str(randint(1000, 100000000))
uploaded_file = st.file_uploader(
    "Choose a file", accept_multiple_files=True, key=state.widget_key)
if st.button('clear uploaded_file'):
    state.widget_key = str(randint(1000, 100000000))
state.sync()


def persist(key: str) -> str:
    """Mark widget state as persistent."""
    if _PERSIST_STATE_KEY not in _state:
        _state[_PERSIST_STATE_KEY] = set()

    _state[_PERSIST_STATE_KEY].add(key)

    return key


def load_widget_state():
    """Load persistent widget state."""
    if _PERSIST_STATE_KEY in _state:
        _state.update(
            {
                key: value
                for key, value in _state.items()
                if key in _state[_PERSIST_STATE_KEY]
            }
        )
