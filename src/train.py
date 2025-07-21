import torch
import torch.optim as optim
import os
from torchvision.utils import save_image
import torch.nn.functional as F
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.utils.data_loader import get_dataloader
from src.utils.losses import ContentLoss, PerceptualLoss, DiscriminatorLoss, GeneratorAdversarialLoss
from src.config import Config

def train_srgan():
    config = Config()
    device = config.DEVICE
    if str(device) == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(config.GENERATED_IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.GENERATOR_CHECKPOINT), exist_ok=True)

    generator = Generator(in_channels=config.IN_CHANNELS, num_rrdb=config.NUM_RRDB_BLOCKS).to(device)
    discriminator = Discriminator(in_channels=config.IN_CHANNELS).to(device)
    if torch.__version__ >= "2.0":
        generator = torch.compile(generator)
        discriminator = torch.compile(discriminator)

    if os.path.exists(config.GENERATOR_CHECKPOINT):
        generator.load_state_dict(torch.load(config.GENERATOR_CHECKPOINT, map_location=device))
    if os.path.exists(config.DISCRIMINATOR_CHECKPOINT):
        discriminator.load_state_dict(torch.load(config.DISCRIMINATOR_CHECKPOINT, map_location=device))

    content_criterion = ContentLoss().to(device)
    perceptual_criterion = PerceptualLoss(vgg_layer=config.VGG_LAYER_FOR_LOSS, device=device).to(device)
    discriminator_criterion = DiscriminatorLoss()
    generator_adversarial_criterion = GeneratorAdversarialLoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE_G, betas=(config.BETA1, config.BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE_D, betas=(config.BETA1, config.BETA2))
    
    scaler_G = torch.cuda.amp.GradScaler(enabled=str(device)=="cuda")
    scaler_D = torch.cuda.amp.GradScaler(enabled=str(device)=="cuda")

    train_dataloader = get_dataloader(
        lr_dir=config.TRAIN_LR_DIR, hr_dir=config.TRAIN_HR_DIR,
        batch_size=config.BATCH_SIZE, shuffle=True,
        scale_factor=config.SCALE_FACTOR, num_workers=2
    )
    sample_dataloader = get_dataloader(
        lr_dir=config.TRAIN_LR_DIR, hr_dir=config.TRAIN_HR_DIR,
        batch_size=1, shuffle=True
    )
    sample_iterator = iter(sample_dataloader)

    print(f"🚀 Training started on {device}")

    for epoch in range(config.NUM_EPOCHS):
        generator.train()
        discriminator.train()
        for i, (lr_images, hr_images) in enumerate(train_dataloader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=str(device)=="cuda"):
                optimizer_D.zero_grad()
                real_preds = discriminator(hr_images)
                sr_images = generator(lr_images).detach()
                fake_preds = discriminator(sr_images)
                d_loss = discriminator_criterion(real_preds, fake_preds)
            
            scaler_D.scale(d_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=str(device)=="cuda"):
                optimizer_G.zero_grad()
                sr_images_g = generator(lr_images)
                fake_preds_g = discriminator(sr_images_g)
                c_loss = content_criterion(sr_images_g, hr_images)
                p_loss = perceptual_criterion(sr_images_g, hr_images)
                adv_loss = generator_adversarial_criterion(fake_preds_g)
                g_loss = c_loss + p_loss + config.LAMBDA_ADVERSARIAL * adv_loss
            
            scaler_G.scale(g_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            
            if (i + 1) % config.LOG_INTERVAL == 0:
                print(f"E[{epoch+1}/{config.NUM_EPOCHS}] S[{i+1}/{len(train_dataloader)}] G_Loss:{g_loss.item():.4f} D_Loss:{d_loss.item():.4f}")

        torch.save(generator.state_dict(), config.GENERATOR_CHECKPOINT)
        torch.save(discriminator.state_dict(), config.DISCRIMINATOR_CHECKPOINT)
        print(f"✅ Models saved at epoch {epoch+1}")
        
        generator.eval()
        with torch.no_grad():
            try:
                sample_lr, sample_hr = next(sample_iterator)
            except StopIteration:
                sample_iterator = iter(sample_dataloader)
                sample_lr, sample_hr = next(sample_iterator)
            
            sample_lr = sample_lr.to(device)
            sr_output = generator(sample_lr).cpu().clamp(-1, 1)
            sr_output = (sr_output + 1) / 2
            
            save_image(sr_output, os.path.join(config.GENERATED_IMAGES_DIR, f"epoch_{epoch+1}_sample.png"))

    print("🏁 Finished Training!")

if __name__ == "__main__":
    train_srgan()
