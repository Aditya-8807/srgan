import torch

class Config:
    # --- Data Paths ---
    # Using the correct paths we discovered from debugging
    DATA_ROOT = './data/DIV2K/'
    TRAIN_HR_DIR = DATA_ROOT + 'DIV2K_train_HR/'
    TRAIN_LR_DIR = DATA_ROOT + 'DIV2K_train_LR_bicubic/X4/'

    # --- Model Parameters ---
    SCALE_FACTOR = 4
    IN_CHANNELS = 3
    NUM_RRDB_BLOCKS = 23

    # --- Training Parameters ---
    BATCH_SIZE = 16      # Suitable for most Colab GPUs
    NUM_EPOCHS = 10      # Increased for better results
    LEARNING_RATE_G = 1e-4
    LEARNING_RATE_D = 2e-4
    BETA1 = 0.9
    BETA2 = 0.999
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 1

    # --- Loss Weights ---
    LAMBDA_ADVERSARIAL = 1e-3
    VGG_LAYER_FOR_LOSS = 'features.35'

    # --- Device Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- File Paths ---
    GENERATOR_CHECKPOINT = './checkpoints/generator.pth'
    DISCRIMINATOR_CHECKPOINT = './checkpoints/discriminator.pth'
    GENERATED_IMAGES_DIR = './results/generated_images/'
    LOGS_DIR = './results/logs/'
