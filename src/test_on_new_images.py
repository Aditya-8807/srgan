import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from src.models.generator import Generator
from src.config import Config

def load_image(image_path, device, img_size=None):
    transform = transforms.Compose([
        transforms.Resize(img_size) if img_size else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def generate_sr_images_from_lr_folder(lr_folder, output_folder, checkpoint_path, config):
    device = config.DEVICE
    generator = Generator(in_channels=config.IN_CHANNELS, num_rrdb=config.NUM_RRDB_BLOCKS).to(device)
    
    # Load checkpoint 60
    checkpoint = torch.load(checkpoint_path, map_location=device ,weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    os.makedirs(output_folder, exist_ok=True)
    image_filenames = [f for f in os.listdir(lr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_filenames:
        img_path = os.path.join(lr_folder, img_file)
        lr_tensor = load_image(img_path, device)  # Shape: [1, 3, H, W]

        with torch.no_grad():
            sr_tensor = generator(lr_tensor)

        sr_tensor = (sr_tensor + 1) / 2  # Denormalize from [-1, 1] → [0, 1]
        save_image(sr_tensor, os.path.join(output_folder, f"SR_{img_file}"))
        print(f"✅ Generated: SR_{img_file}")

if __name__ == "__main__":
    config = Config()
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_epoch_60.pth")
    lr_folder = "test_images/LR"  # Your test LR images
    output_folder = "test_images/SR"

    generate_sr_images_from_lr_folder(lr_folder, output_folder, checkpoint_path, config)