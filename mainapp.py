import streamlit as st

from pages.chatbot import *

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
