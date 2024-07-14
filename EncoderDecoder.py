import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

class BlackWhiteDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(512 * 16 * 16, 2048),
            nn.ReLU(),
        )
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(2048, 512 * 16 * 16),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Sigmoid to normalize output to [0, 1]
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten for the linear layer
        x = self.encoder_fc(x)
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 512, 16, 16)  # Reshape to the size before entering transposed conv layers
        x = self.decoder_conv(x)
        return x

def create_data_loader(folder_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    dataset = BlackWhiteDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader

def train(model, criterion, optimizer, dataloader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, imgs in enumerate(dataloader, 0):
            imgs = imgs.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, imgs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

                # Visualize reconstruction of some input images
                num_images = 5
                plt.figure(figsize=(10, 2))
                for j in range(num_images):
                    plt.subplot(2, num_images, j + 1)
                    plt.imshow(np.transpose(imgs[j].cpu().numpy(), (1, 2, 0)), cmap='gray')
                    plt.title('Original')
                    plt.axis('off')
                    
                    plt.subplot(2, num_images, num_images + j + 1)
                    plt.imshow(np.transpose(outputs[j].detach().cpu().numpy(), (1, 2, 0)), cmap='gray')
                    plt.title('Reconstruction')
                    plt.axis('off')
                plt.show()


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Display the output
    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            break

dataloader = create_data_loader(r"/Users/nikitadrozdov/Downloads/images")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
model = Autoencoder().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10

train(model, criterion, optimizer, dataloader, num_epochs)