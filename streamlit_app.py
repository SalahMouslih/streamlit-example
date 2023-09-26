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

uploaded_file = st.file_uploader("Upload your file here...", type=['csv'])

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)

	X = dataframe[["column1","column2"]]
	result = trained_model.predict(X)

	st.write(f"Your prediction is: {result}")
