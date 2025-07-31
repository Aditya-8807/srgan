import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import Config

def moving_average(data, window_size=5):
    """Compute moving average to smooth data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results():
    config = Config()

    if not os.path.exists(config.HISTORY_FILE):
        print(f"❌ History file not found at {config.HISTORY_FILE}")
        return

    print("=> Loading history to plot graphs...")
    history = np.load(config.HISTORY_FILE)

    g_loss_history = history['g_loss']
    d_loss_history = history['d_loss']
    psnr_history = history['psnr']
    epochs_ran = int(history['epoch'])

    if epochs_ran == 0:
        print("No history found. Please train for at least one epoch.")
        return

    # --- Plot 1: Training Losses ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_ran + 1), g_loss_history, label='Generator Loss', color='blue')
    plt.plot(range(1, epochs_ran + 1), d_loss_history, label='Discriminator Loss', color='red')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)

    # --- Plot 2: PSNR with smoothing and annotation ---
    plt.subplot(1, 2, 2)

    # Smooth the PSNR curve
    smoothed_psnr = moving_average(psnr_history, window_size=5)
    smoothed_epochs = range(5, epochs_ran + 1)

    plt.plot(smoothed_epochs, smoothed_psnr, label='PSNR ', color='green')

    # Mark the max PSNR point
    max_epoch = np.argmax(psnr_history) + 1
    max_psnr = psnr_history[max_epoch - 1]
    plt.axvline(max_epoch, color='gray', linestyle='--', linewidth=1)
    plt.text(max_epoch, max_psnr, f'Max PSNR: {max_psnr:.2f} dB\n@ Epoch {max_epoch}',
             fontsize=8, color='black', ha='right')

    plt.title('PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    # --- Save and Show ---
    plt.tight_layout()
    plot_path = os.path.join(config.LOGS_DIR, 'performance_graphs.png')
    plt.savefig(plot_path)
    print(f"✅ Graphs saved to {plot_path}")
    plt.show()

if __name__ == '__main__':
    plot_results()