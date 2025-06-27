# Google Colab setup
!pip install torchvision matplotlib --quiet

## Dataset Preparation

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

# Create HR and LR transforms
hr_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

lr_transform = transforms.Compose([
    transforms.Resize((24, 24)),
    transforms.Resize((96, 96)),  # Upsample to match HR size
    transforms.ToTensor()
])

# Custom dataset class
class SRDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, hr_transform, lr_transform):
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.jpg')]
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index]).convert('RGB')
        hr = self.hr_transform(img)
        lr = self.lr_transform(img)
        return lr, hr

    def __len__(self):
        return len(self.image_filenames)

# Example usage
image_dir = "/content/data"  # Upload your dataset here
dataset = SRDataset(image_dir, hr_transform, lr_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)


##  Define SRGAN Generator and Discriminator

import torch.nn as nn

# Simple Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# Generator (simplified for 4x upscaling)
class Generator(nn.Module):
    def __init__(self, num_residuals=16):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residuals)])

        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),

            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.output = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.residuals(x1)
        x2 = self.bn_mid(self.conv_mid(x2))
        x = x1 + x2
        x = self.upsample(x)
        return self.output(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c, stride=1, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(3, 64, bn=False),
            block(64, 64, stride=2),
            block(64, 128),
            block(128, 128, stride=2),
            block(128, 256),
            block(256, 256, stride=2),
            block(256, 512),
            block(512, 512, stride=2),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

## Loss and Training Logic

import torch.optim as optim
from torchvision.models import vgg19

# Instantiate networks
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# Losses
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

# VGG-based perceptual loss
vgg = vgg19(pretrained=True).features[:36].eval().cuda()
for param in vgg.parameters():
    param.requires_grad = False

def perceptual_loss(sr, hr):
    return mse_loss(vgg(sr), vgg(hr))

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

## Training Loop

from tqdm import tqdm

epochs = 100

for epoch in range(epochs):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for lr_imgs, hr_imgs in loop:
        lr_imgs, hr_imgs = lr_imgs.cuda(), hr_imgs.cuda()
        
        # Train Discriminator
        fake_imgs = generator(lr_imgs)
        real_labels = torch.ones(lr_imgs.size(0), 1).cuda()
        fake_labels = torch.zeros(lr_imgs.size(0), 1).cuda()

        optimizer_D.zero_grad()
        real_loss = bce_loss(discriminator(hr_imgs), real_labels)
        fake_loss = bce_loss(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        adversarial_loss = bce_loss(discriminator(fake_imgs), real_labels)
        content_loss = perceptual_loss(fake_imgs, hr_imgs)
        g_loss = content_loss + 1e-3 * adversarial_loss
        g_loss.backward()
        optimizer_G.step()

        loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

## Saving the Model

torch.save(generator.state_dict(), "srgan_generator.pth")
torch.save(discriminator.state_dict(), "srgan_discriminator.pth")

