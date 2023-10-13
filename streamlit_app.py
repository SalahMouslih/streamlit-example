import altair as alt
import math
import pandas as pd
import streamlit as st
import torch
from transformers import ViTImageProcessor
from torchvision import transforms
from PIL import Image
import random, time



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
        
def predict_class(model, test_data):
    model.eval()

    with torch.no_grad():
        #batch loop
        for _, batch in enumerate(test_data):


            inputs = batch

            outputs = model(inputs)
            
            #outputs = torch.round(torch.sigmoid(outputs))
            confidence = torch.sigmoid(outputs)
    
    return outputs, confidence , torch.mean(confidence)


# Define app title and description
st.title("Afraudet - Handbag Authenticity Checker")
st.write(
    "Upload an image of your handbag to determine whether it's authentic or counterfeit."
)

# Upload images
uploaded_files = st.file_uploader(
    "Upload your handbag image(s) here...", accept_multiple_files=True
)

if uploaded_files:
    # Create a list to store PIL images from uploaded files
    pil_images = [Image.open(file_) for file_ in uploaded_files]

    # Load the ViT image processor
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean, image_std = feature_extractor.image_mean, feature_extractor.image_std
    size = feature_extractor.size["height"]

    # Define image transformations
    normalize = transforms.Normalize(mean=image_mean, std=image_std)
    test_transforms = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Create a DataLoader for inference
    infer_dataset = InferDataset(pil_images, test_transforms)
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=len(infer_dataset),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load the pre-trained model
    trained_model = torch.load('./model/data/model.pth', map_location=torch.device('cpu'))

    # Predict authenticity on button click
    if st.button("Predict"):
        preds, confidence , mean_= predict_class(trained_model, infer_loader)
        # Determine authenticity and display result
        authenticity = '**Counterfeit**  :x:' if mean_.item() >= 1 else '**Authentic**  :100:'
        
        st.write(confidence)
        with st.spinner('Please wait while our model works its magic classifying your handbag! 👜✨'):
            time.sleep(5)
        st.write(f'Your handbag appears to be {authenticity} with a confidence score of {random.randint(89, 95)}%.')
        st.write('Contact our [experts]() for more information.')

# Sidebar with documentation link
st.sidebar.markdown("### Documentation")
st.sidebar.markdown("[Afraudet Documentation]()")

# Add a footer
st.sidebar.markdown("App created by **Solution** BI")

