from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import mlflow

%pip install mlflow

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

import mlflow
logged_model = 'runs:/3c802963efed4b84be2eea9ccb96be8a/pytorch-tuned-EFF_NET-model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

if loaded_model:
  print('success')
else : print('nothing')
# Predict on a Pandas DataFrame.
#loaded_model.predict(pd.DataFrame(data))
