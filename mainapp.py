import streamlit as st

from pages.chatbot import Chatbot_app
from pages.info import Info_app

PAGES = {
    "Info": Info_app,
    "Chatbot": Chatbot_app
}


st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))


# Select the page
page = PAGES[selection]

# Provide the API key to the page function
page()
