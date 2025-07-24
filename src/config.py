import torch

class Config:
    # --- Data Paths ---
    DATA_ROOT = './data/DIV2K/'
    TRAIN_HR_DIR = DATA_ROOT + 'DIV2K_train_HR/'
    TRAIN_LR_DIR = DATA_ROOT + 'DIV2K_train_LR_bicubic/X4/'

    # --- Model & Training Parameters ---
    SCALE_FACTOR = 4
    IN_CHANNELS = 3
    NUM_RRDB_BLOCKS = 23
    BATCH_SIZE = 32
    NUM_EPOCHS = 100      # You set this to 20
    LEARNING_RATE_G = 1e-4
    LEARNING_RATE_D = 2e-4
    BETA1 = 0.9
    BETA2 = 0.999
    LOG_INTERVAL = 10

    SAVE_INTERVAL = 20 # Save every 20 epochs

    # Add this line back!
    SAVE_EPOCHS = [] # You can leave this empty if you only want to save every SAVE_INTERVAL epochs
                     # Or populate it with specific epochs like [10, 50, 100, 150, 200]
                     # for additional checkpoints.

    # --- Loss Weights ---
    LAMBDA_ADVERSARIAL = 1e-3
    VGG_LAYER_FOR_LOSS = 'features.35'

    # --- Device Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths for Checkpoints, History, and Results ---
    CHECKPOINT_DIR = './checkpoints/'
    HISTORY_FILE = './checkpoints/training_history.npz'
    GENERATED_IMAGES_DIR = './results/generated_images/'
    LOGS_DIR = './results/logs/'

    # --- Paths for Image Generation ---
    GENERATE_LR_DIR = './generate/low_res/'
    GENERATE_HR_DIR = './generate/high_res/'
