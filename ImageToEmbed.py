from EncoderDecoder import Autoencoder
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

class ImageToEmbed():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = Autoencoder().to(self.device)
        self.model.load_state_dict(torch.load("1000emb.pt"))
        self.model.eval()

    def convert(self, image_path):

        # Load the image and apply transformations
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        image = Image.open(image_path).convert('L')
        image = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        # Pass the image through the encoder to get the embedding
        with torch.no_grad():
            embedding = self.model.get_embed(image)


        return embedding.cpu().numpy()[0]
