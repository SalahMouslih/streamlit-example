from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import mlflow


"""
# Welcome to Afraudet!

Please

Refe to [documentation](https://docs.streamlit.io) of the app to understand app components

"""


"""
### Classify bags
"""

# load pre-trained model
trained_model = pickle.load(open(model/data/model.pth, 'rb'))

uploaded_files = st.file_uploader("Upload your files here...", accept_multiple_files=True)


if trained_model:
  print('success')
else : print('nothing')


# Predict on a Pandas DataFrame.
#loaded_model.predict(pd.DataFrame(data))
