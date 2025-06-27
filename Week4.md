# Week 4 - Annotated Heatmap Visualization (e.g., for Loss, PSNR, SSIM)
# 🔍 Useful to visualize training/validation results across epochs and models

import matplotlib.pyplot as plt
import numpy as np

# ✅ Utility to create annotated heatmap
def plot_annotated_heatmap(data, row_labels, col_labels, title="", cmap="YlGnBu"):
    fig, ax = plt.subplots(figsize=(len(col_labels) + 2, len(row_labels) + 2))
    im = ax.imshow(data, cmap=cmap)

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate tick labels and align
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Annotate each cell with the value
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f"{data[i, j]:.2f}",
                           ha="center", va="center", color="black")

    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()

# ✅ Example data for training results
# Rows = Models, Columns = Epochs
example_data = np.array([
    [25.3, 26.1, 26.8, 27.4],   # SRGAN
    [26.5, 27.6, 28.1, 28.9],   # ESRGAN
])

models = ["SRGAN", "ESRGAN"]
epochs = ["Epoch 10", "Epoch 20", "Epoch 30", "Epoch 40"]

# 🖼️ Plot the heatmap
plot_annotated_heatmap(example_data, models, epochs, title="PSNR Over Epochs")