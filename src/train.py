import torch
import torch.optim as optim
import os
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.utils.data_loader import get_dataloader
from src.utils.losses import ContentLoss, PerceptualLoss, DiscriminatorLoss, GeneratorAdversarialLoss
from src.utils.metrics import calculate_psnr
from src.config import Config

def train_srgan():
    # --- 1. INITIAL SETUP ---
    config = Config()
    device = config.DEVICE
    start_epoch = 0
    best_psnr = -1.0 # Initialize with a low value for PSNR to ensure first model is saved

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.GENERATED_IMAGES_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # --- 2. INITIALIZE MODELS AND OPTIMIZERS ---
    generator = Generator(in_channels=config.IN_CHANNELS, num_rrdb=config.NUM_RRDB_BLOCKS).to(device)
    discriminator = Discriminator(in_channels=config.IN_CHANNELS).to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE_G, betas=(config.BETA1, config.BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE_D, betas=(config.BETA1, config.BETA2))

    g_loss_history, d_loss_history, psnr_history = [], [], []

    # --- 3. RESUME FROM CHECKPOINT LOGIC ---
    # Find the latest checkpoint (more robust than relying solely on history.npz)
    latest_checkpoint_epoch = -1
    latest_checkpoint_path = None
    for f in os.listdir(config.CHECKPOINT_DIR):
        if f.startswith("checkpoint_epoch_") and f.endswith(".pth"):
            try:
                epoch_num = int(f.replace("checkpoint_epoch_", "").replace(".pth", ""))
                if epoch_num > latest_checkpoint_epoch:
                    latest_checkpoint_epoch = epoch_num
                    latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f)
            except ValueError:
                continue

    if latest_checkpoint_path:
        print(f"=> Loading checkpoint from {latest_checkpoint_path}...")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
            g_loss_history = list(checkpoint['g_loss_history'])
            d_loss_history = list(checkpoint['d_loss_history'])
            psnr_history = list(checkpoint['psnr_history'])
            best_psnr = checkpoint.get('best_psnr', -1.0) # Retrieve best_psnr if saved
            print(f"✅ Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"❌ Error loading checkpoint {latest_checkpoint_path}: {e}")
            print("Starting training from scratch.")
            start_epoch = 0
            g_loss_history, d_loss_history, psnr_history = [], [], []
            best_psnr = -1.0 # Reset best_psnr if starting fresh
    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- 4. LOSS FUNCTIONS AND DATALOADERS ---
    content_criterion = ContentLoss().to(device)
    perceptual_criterion = PerceptualLoss(vgg_layer=config.VGG_LAYER_FOR_LOSS, device=device)
    discriminator_criterion = DiscriminatorLoss()
    generator_adversarial_criterion = GeneratorAdversarialLoss()

    train_dataloader = get_dataloader(
        lr_dir=config.TRAIN_LR_DIR, hr_dir=config.TRAIN_HR_DIR,
        batch_size=config.BATCH_SIZE, shuffle=True,
        scale_factor=config.SCALE_FACTOR, num_workers=2
    )
    # Using a separate dataloader for sample image to avoid issues with batching
    # when you only need one image.
    sample_dataloader = get_dataloader(
        lr_dir=config.TRAIN_LR_DIR, hr_dir=config.TRAIN_HR_DIR,
        batch_size=1, shuffle=True,
        scale_factor=config.SCALE_FACTOR, num_workers=1 # Using 1 worker for single image
    )
    sample_iterator = iter(sample_dataloader)

    print(f"🚀 Starting training from epoch {start_epoch}...")

    # --- 5. MAIN TRAINING LOOP ---
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        generator.train()
        discriminator.train()

        for i, (lr_images, hr_images) in enumerate(train_dataloader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            real_preds = discriminator(hr_images)
            sr_images = generator(lr_images).detach() # Detach to prevent gradients flowing to G
            fake_preds = discriminator(sr_images)
            d_loss = discriminator_criterion(real_preds, fake_preds)
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            sr_images_g = generator(lr_images)
            fake_preds_g = discriminator(sr_images_g)
            c_loss = content_criterion(sr_images_g, hr_images)
            p_loss = perceptual_criterion(sr_images_g, hr_images)
            adv_loss = generator_adversarial_criterion(fake_preds_g)
            g_loss = c_loss + p_loss + config.LAMBDA_ADVERSARIAL * adv_loss
            g_loss.backward()
            optimizer_G.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            if (i + 1) % config.LOG_INTERVAL == 0:
                print(f"E[{epoch+1}/{config.NUM_EPOCHS}] S[{i+1}/{len(train_dataloader)}] G_Loss:{g_loss.item():.4f} D_Loss:{d_loss.item():.4f}")

        # --- 6. END OF EPOCH: CALCULATE METRICS AND SAVE ---
        avg_g_loss = epoch_g_loss / len(train_dataloader)
        avg_d_loss = epoch_d_loss / len(train_dataloader)
        g_loss_history.append(avg_g_loss)
        d_loss_history.append(avg_d_loss)
        print(f"End of Epoch {epoch+1} | Avg. G_Loss: {avg_g_loss:.4f} | Avg. D_Loss: {avg_d_loss:.4f}")

        generator.eval()
        with torch.no_grad():
            try:
                sample_lr, sample_hr = next(sample_iterator)
            except StopIteration: # Reset iterator if exhausted
                sample_iterator = iter(sample_dataloader)
                sample_lr, sample_hr = next(sample_iterator)

            sample_lr, sample_hr = sample_lr.to(device), sample_hr.to(device)
            sr_output = generator(sample_lr)

            # Denormalize images for PSNR calculation and saving
            sr_unnorm = (sr_output + 1) / 2
            hr_unnorm = (sample_hr + 1) / 2
            current_psnr = calculate_psnr(sr_unnorm, hr_unnorm, data_range=1.0)
            psnr_history.append(current_psnr)
            print(f"PSNR on sample image: {current_psnr:.4f} dB")

            save_image(sr_unnorm.cpu(), os.path.join(config.GENERATED_IMAGES_DIR, f"epoch_{epoch+1}_sample.png"))

        # --- Checkpoint Saving Logic ---
        # 1. Save every `SAVE_INTERVAL` epochs
        # 2. Save at specific `SAVE_EPOCHS` (from config)
        # 3. Always save the "best" model based on PSNR
        # 4. Always save the history.npz file (as it's used by plot_results)

        # Determine if we should save a periodic checkpoint
        should_save_periodic = (epoch + 1) % config.SAVE_INTERVAL == 0

        # Determine if we should save a specific checkpoint from the list
        should_save_specific = (epoch + 1) in config.SAVE_EPOCHS

        # Save a full checkpoint (models, optimizers, history)
        if should_save_periodic or should_save_specific:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"✅ Saving checkpoint for epoch {epoch+1}...")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_loss_history': g_loss_history,
                'd_loss_history': d_loss_history,
                'psnr_history': psnr_history,
                'best_psnr': best_psnr, # Save current best_psnr
            }, checkpoint_path)

        # Save the best model based on PSNR (validation PSNR is usually better here)
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            print(f"🌟 New best model found! Saving to {best_model_path} with PSNR: {best_psnr:.4f}")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'best_psnr': best_psnr,
            }, best_model_path) # Only save generator and best_psnr for simplicity of "best model"

        # Always save the history file to track progress for resuming plot_results.py
        np.savez(config.HISTORY_FILE,
                 epoch=epoch + 1,
                 g_loss=g_loss_history,
                 d_loss=d_loss_history,
                 psnr=psnr_history)

    print("🏁 Finished Training!")

if __name__ == "__main__":
    train_srgan()
