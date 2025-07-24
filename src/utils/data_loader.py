import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.image_pairs = []

        lr_filenames = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        for lr_fn in lr_filenames:
            hr_fn = lr_fn.replace('x4', '')
            hr_path = os.path.join(self.hr_dir, hr_fn)
            if os.path.exists(hr_path):
                self.image_pairs.append((lr_fn, hr_fn))

        if not self.image_pairs:
            raise FileNotFoundError(f"No matching image pairs found between {lr_dir} and {hr_dir}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        lr_fn, hr_fn = self.image_pairs[idx]
        lr_path = os.path.join(self.lr_dir, lr_fn)
        hr_path = os.path.join(self.hr_dir, hr_fn)
        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")
        if self.transform:
            lr_image = self.transform['lr'](lr_image)
            hr_image = self.transform['hr'](hr_image)
        return lr_image, hr_image

def get_dataloader(lr_dir, hr_dir, batch_size, shuffle, scale_factor=4, num_workers=2, pin_memory=True):
    HR_IMAGE_SIZE = 256
    LR_IMAGE_SIZE = HR_IMAGE_SIZE // scale_factor
    hr_transform = transforms.Compose([
        transforms.Resize((HR_IMAGE_SIZE, HR_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    lr_transform = transforms.Compose([
        transforms.Resize((LR_IMAGE_SIZE, LR_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_dict = {'hr': hr_transform, 'lr': lr_transform}
    dataset = SRDataset(lr_dir=lr_dir, hr_dir=hr_dir, transform=transform_dict)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
