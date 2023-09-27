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
from torch.utils.data import Dataset
from PIL import Image


"""
# Welcome to Afraudet!

Please Refer to [documentation](https://docs.streamlit.io) to understand app components

"""


"""
### Classify bags
Upload your images here
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


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, pil_imgs, _test_transforms):
        super(InferDataset, self,).__init__()

        self.pil_imgs = pil_imgs
        self.transform = _test_transforms

    def __len__(self):
        return len(self.pil_imgs)

    def __getitem__(self, idx):
        img = self.pil_imgs[idx]

        return self.transform(img)


# load pre-trained model
trained_model = torch.load('./model/data/model.pth',map_location=torch.device('cpu') )

uploaded_files = st.file_uploader("Upload your files here...", accept_multiple_files=True)

pil_images = [Image.open(file_) for file_ in uploaded_files]


image_dataset = InferDataset(pil_images,_test_transforms)
infer_loader = torch.utils.data.DataLoader(image_dataset,
                                           batch_size=len(image_dataset),
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True)

if st.button("Predict"):
    preds = predict_class(infer_loader)
    st.write("Prediction:", preds)
    

