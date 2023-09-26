from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import torch, torchvision
from transformers import ViTImageProcessor
from torchvision import transforms

"""
# Welcome to Afraudet!

Please

Refe to [documentation](https://docs.streamlit.io) of the app to understand app components

"""


"""
### Classify bags
"""

_test_transforms = transforms.Compose(
    [
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ]
)


# load pre-trained model
trained_model = torch.load('./model/data/model.pth',map_location=torch.device('cpu') )

uploaded_files = st.file_uploader("Upload your files here...", accept_multiple_files=True)

test_data = torchvision.datasets.ImageFolder(uploaded_files, transform = _test_transforms)

def predict_class():
    return trained_model.predict(test_data)

if st.button("Predict"):

    predict_class()

print(trained_model)

# Predict on a Pandas DataFrame.
#loaded_model.predict(pd.DataFrame(data))
