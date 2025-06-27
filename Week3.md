# Week 3 - PyTorch DataLoader and Custom Dataset Pipeline for SR Tasks

# ✅ Objective:
# Build a flexible dataset loader for low-resolution (LR) and high-resolution (HR) image pairs
# Use with SRGAN, ESRGAN, etc. for training

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 🔧 Configurable transforms for HR and LR image creation
def get_sr_transforms(hr_size=96, scale=4):
    lr_size = hr_size // scale
    hr_transform = transforms.Compose([
        transforms.Resize((hr_size, hr_size)),
        transforms.ToTensor()
    ])

    lr_transform = transforms.Compose([
        transforms.Resize((lr_size, lr_size)),
        transforms.Resize((hr_size, hr_size)),  # Upsample back for model input
        transforms.ToTensor()
    ])
    return hr_transform, lr_transform

# 📁 Custom Dataset Class for SR
class SuperResolutionDataset(Dataset):
    def __init__(self, image_dir, hr_transform, lr_transform):
        super().__init__()
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        hr = self.hr_transform(img)
        lr = self.lr_transform(img)
        return lr, hr

    def __len__(self):
        return len(self.image_paths)

# ⚙️ Function to get DataLoader

def get_dataloader(image_dir, hr_size=96, scale=4, batch_size=4, shuffle=True, num_workers=2):
    hr_transform, lr_transform = get_sr_transforms(hr_size, scale)
    dataset = SuperResolutionDataset(image_dir, hr_transform, lr_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# ✅ Example Usage:
# image_dir = "/content/data"  # Path to HR images (LR will be generated on the fly)
# dataloader = get_dataloader(image_dir)
# for lr, hr in dataloader:
#     print(lr.shape, hr.shape)  # Expected: torch.Size([B, 3, 96, 96])
