from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import torch

"""
# Welcome to Afraudet!

Please

Refe to [documentation](https://docs.streamlit.io) of the app to understand app components

"""


"""
### Classify bags
"""

def transform(mode):
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """

    from transformers import ViTImageProcessor
    from torchvision import transforms

    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    image_mean, image_std = feature_extractor.image_mean, feature_extractor.image_std
    size = feature_extractor.size["height"]
    normalize = transforms.Normalize(mean=image_mean, std=image_std)

    _train_transforms = transforms.Compose(
        [   
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    _val_transforms = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    _test_transforms = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    transforms =  {'train': _train_transforms, 'val': _val_transforms, 'test': _test_transforms}

    return transforms[mode]


# load pre-trained model
trained_model = torch.load('./model/data/model.pth',map_location=torch.device('cpu') )

uploaded_files = st.file_uploader("Upload your files here...", accept_multiple_files=True)

test_data = torchvision.datasets.ImageFolder(uploaded_files, transform = transforms('test'))

if st.button(“Predict”):

    predict_class()

print(trained_model)

# Predict on a Pandas DataFrame.
#loaded_model.predict(pd.DataFrame(data))
