# test_on_new_images.py
import os
import torch
from PIL import Image
from torchvision import transforms
from src.models.generator import Generator
from src.config import Config

# Load model
config = Config()
device = config.DEVICE
generator = Generator().to(device)
checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pth"), map_location=device ,weights_only=False)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# Image transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
inv_transform = transforms.Normalize(
    mean=[-1, -1, -1],
    std=[2, 2, 2]
)

# Inference
input_folder = './test_images/lr/'
output_folder = './test_images/sr/'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(os.path.join(input_folder, filename)).convert("RGB")
        lr_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = generator(lr_tensor)
        
        sr_tensor = (sr_tensor + 1) / 2  # Denormalize to [0,1]
        save_path = os.path.join(output_folder, filename)
        transforms.ToPILImage()(sr_tensor.squeeze().cpu()).save(save_path)
        print(f"✅ Saved SR image: {save_path}")