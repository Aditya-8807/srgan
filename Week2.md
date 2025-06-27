# ESRGAN Week 2 - Clean PyTorch Implementation (Colab-Compatible)

# Step 1: Install Requirements
!pip install torchvision matplotlib --quiet

# Step 2: Reuse Dataset (Same as Week 1)
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

hr_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

lr_transform = transforms.Compose([
    transforms.Resize((24, 24)),
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

class SRDataset(Dataset):
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

# Step 3: Define RRDB Block and ESRGAN Generator
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(5):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_channels, growth_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return 0.2 * features[-1] + x

class RRDB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            DenseBlock(in_channels),
            DenseBlock(in_channels),
            DenseBlock(in_channels)
        )

    def forward(self, x):
        return self.block(x) * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_feat=64, num_blocks=23):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        self.rrdb_blocks = nn.Sequential(*[RRDB(num_feat) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        fea = self.initial(x)
        trunk = self.trunk_conv(self.rrdb_blocks(fea))
        fea = fea + trunk
        out = self.upsample(fea)
        return self.final(out)

# Step 4: Define Discriminator (same as SRGAN)
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

# Step 5: Losses and Optimizers
from torchvision.models import vgg19
import torch.optim as optim

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

vgg = vgg19(pretrained=True).features[:36].eval().cuda()
for p in vgg.parameters():
    p.requires_grad = False

def perceptual_loss(sr, hr):
    return mse_loss(vgg(sr), vgg(hr))

G = RRDBNet().cuda()
D = Discriminator().cuda()

optimizer_G = optim.Adam(G.parameters(), lr=1e-4)
optimizer_D = optim.Adam(D.parameters(), lr=1e-4)

# Step 6: Training Loop
from tqdm import tqdm

def train_esrgan(dataloader, epochs):
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for lr_imgs, hr_imgs in loop:
            lr_imgs, hr_imgs = lr_imgs.cuda(), hr_imgs.cuda()
            fake_imgs = G(lr_imgs)
            real_labels = torch.ones(lr_imgs.size(0), 1).cuda()
            fake_labels = torch.zeros(lr_imgs.size(0), 1).cuda()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = bce_loss(D(hr_imgs), real_labels)
            fake_loss = bce_loss(D(fake_imgs.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            adv_loss = bce_loss(D(fake_imgs), real_labels)
            content_loss = perceptual_loss(fake_imgs, hr_imgs)
            g_loss = content_loss + 1e-3 * adv_loss
            g_loss.backward()
            optimizer_G.step()

            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

# Save weights
# torch.save(G.state_dict(), "esrgan_generator.pth")
# torch.save(D.state_dict(), "esrgan_discriminator.pth")
