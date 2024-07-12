import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

class BlackWhiteDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name).convert('L')
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        return image, image  # Input and target are the same for autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Sigmoid to normalize output to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_data_loader(folder_path):
    # Create dataset and dataloader
    dataset = BlackWhiteDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader

def train(model, criterion, optimizer, num_epochs):
    
    for epoch in range(num_epochs):
        cnt = 0
        for imgs, _ in dataloader:
            cnt+=1
            print(cnt)
            imgs = imgs.to(device)

            # Forward pass
            outputs = model(imgs)
            print(imgs.shape, outputs.shape)
            loss = criterion(outputs, imgs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Display the output
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            break
dataloader = create_data_loader(r"C:\Users\abdur\Downloads\images")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10

train(model, criterion, optimizer, num_epochs)
