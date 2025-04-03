import torch
import torch.nn as nn
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the same generator architecture
latent_size = 128  # Make sure this matches your training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)

# Load the saved model
generator.load_state_dict(torch.load('G.pth', map_location=device))
generator.to(device)
generator.eval()  # Set to evaluation mode

def generate_images(num_images):
    latent = torch.randn(num_images, latent_size, 1, 1, device=device)  # Random noise
    with torch.no_grad():
        fake_images = generator(latent).cpu().numpy()  # Convert to NumPy
    fake_images = (fake_images + 1) / 2  # Normalize to [0,1]
    
    # Reshape images for display
    fake_images = [np.transpose(img, (1, 2, 0)) for img in fake_images]  # (C, H, W) -> (H, W, C)
    return fake_images

st.set_page_config(page_title="AnimeGAN AAI", page_icon="🎨")
st.title("AnimeGAN Image Generator 🎨")
st.write("Select the number of images to generate and click the button.")

num_images = st.number_input("Number of images:", min_value=1, max_value=10, value=1, step=1)

if st.button("Generate Images"):
    fake_images = generate_images(num_images)
    
    # Display images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    if num_images == 1:
        axes.imshow(fake_images[0])
        axes.axis("off")
    else:
        for ax, img in zip(axes, fake_images):
            ax.imshow(img)
            ax.axis("off")
    st.pyplot(fig)
