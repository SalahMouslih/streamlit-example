from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import torch, torchvision
from transformers import ViTImageProcessor
from torchvision import transforms, datasets
import os

"""
# Welcome to Afraudet!

Please

Refe to [documentation](https://docs.streamlit.io) of the app to understand app components

"""


"""
### Classify bags
"""
def predict_class(test_data):
    return trained_model(test_data)


feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

image_mean, image_std = feature_extractor.image_mean, feature_extractor.image_std
size = feature_extractor.size["height"]

normalize = transforms.Normalize(mean=image_mean, std=image_std)
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

# Check if files were uploaded
if uploaded_files:
    # Create a temporary directory to store the uploaded images
    tmp_dir = "tmp_uploaded_images/inf"
    os.makedirs(tmp_dir, exist_ok=True)

    # Iterate through the uploaded files and save them to the temporary directory
    for i, file in enumerate(uploaded_files):
        with open(os.path.join(tmp_dir, f"image_{i}.jpg"), "wb") as f:
            f.write(file.read())

    """
    print(os.listdir(tmp_dir))
    """

    # Create an ImageFolder dataset from the uploaded images
    #image_dataset = datasets.ImageFolder(root=tmp_dir, transform=_test_transforms)

    if st.button("Predict"):
        preds = predict_class(image_dataset)

        '''
        predictions: {{preds}} 
        '''
    

"""


"""



