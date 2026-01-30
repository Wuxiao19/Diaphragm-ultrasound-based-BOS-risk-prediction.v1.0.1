"""
Streamlit does not support session state natively. This module provides a
SessionState class that can be used to store information across reruns.
"""

import streamlit as st

class _SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __call__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def sync(self):
        for key, val in self.__dict__.items():
            st.session_state[key] = val


def get_session_state(**kwargs):
    session_state = st.session_state.get("_session_state", None)
    if session_state is None:
        session_state = _SessionState(**kwargs)
        st.session_state["_session_state"] = session_state

    return session_state