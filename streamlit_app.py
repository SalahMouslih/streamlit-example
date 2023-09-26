from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd

"""
# Welcome to Afraudet!

Please

Refe to [documentation](https://docs.streamlit.io) of the app to understand app components

"""

# load pre-trained model
#trained_model = pickle.load(open(model_filename, 'rb'))

"""
### Classify bags
"""
uploaded_files = st.file_uploader("Upload your files here...", accept_multiple_files=True)


