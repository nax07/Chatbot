import streamlit as st

from Info import Info_app
from Chatbot import Chatbot_app
from Chat_search import Search_app
from QyA import QyA_app
from LangStart import Lang_app

from libraries.text_translation import *
from libraries.text_processing import *

import streamlit as st

PAGES = {
    "Info": Info_app,
    "Chatbot": Chatbot_app,
    "Chat with Search": Search_app,
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))


# Select the page
page = PAGES[selection]

# Provide the API key to the page function
with st.container():
    page()
