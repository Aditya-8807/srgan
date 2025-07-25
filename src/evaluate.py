import torch
import os
from PIL import Image
from torchvision import transforms
import lpips # Import LPIPS

from src.models.generator import Generator
from src.utils.metrics import calculate_psnr, calculate_ssim
from src.config import Config

# --- SETUP ---
config = Config()
device = config.DEVICE
generator = Generator().to(device)
checkpoint = torch.load('./checkpoints/checkpoint_epoch_100.pth', map_location=device, weights_only=False)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# --- Initialize LPIPS ---
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

# --- Define Transforms ---
# Define transforms to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
])

# --- EVALUATION LOOP ---
test_hr_dir = './benchmark/Set5/HR/'
test_image_files = sorted(os.listdir(test_hr_dir))
total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0

print(f"🧪 Evaluating on {os.path.basename(os.path.dirname(test_hr_dir))} dataset...")

for filename in test_image_files:
    # Load HR image
    hr_img = Image.open(os.path.join(test_hr_dir, filename)).convert("RGB")

    # Create LR image by downscaling
    lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)

    # Convert to tensors
    hr_tensor = transform(hr_img).unsqueeze(0).to(device)
    lr_tensor = transform(lr_img).unsqueeze(0).to(device)

    # Generate SR image
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)

    # --- Calculate Metrics ---
    # Denormalize images from [-1, 1] to [0, 1] for metric calculation
    hr_unnorm = (hr_tensor + 1) / 2
    sr_unnorm = (sr_tensor + 1) / 2

    total_psnr += calculate_psnr(sr_unnorm, hr_unnorm, data_range=1.0)
    total_ssim += calculate_ssim(sr_unnorm, hr_unnorm, data_range=1.0)
    
    # LPIPS expects tensors in [-1, 1] range, so we use the original tensors
    total_lpips += loss_fn_alex(sr_tensor, hr_tensor).item()

# --- Calculate and Print Averages ---
avg_psnr = total_psnr / len(test_image_files)
avg_ssim = total_ssim / len(test_image_files)
avg_lpips = total_lpips / len(test_image_files)

print(f"--- Results ---")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average LPIPS: {avg_lpips:.4f}")