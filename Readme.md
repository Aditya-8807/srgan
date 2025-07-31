I am Aditya Chaurasiya, a sophomore from the Civil Engineering department with a passion for deep learning and computer vision. This project is a part of my journey to explore the exciting applications of AI.Here's repo details of SRGAN project

## Table of Contents

* [About The Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Requirements](#Requirements)
  * [Installation](#installation)
* [Usage](#usage)
  * [ Training the Model](#1-training-the-model)
  * [ Generating New Images](#3-generating-new-images)
* [Project Structure](#project-structure)
* [Results](#results)

## About The Project

This project addresses the common problem of **image super-resolution**. Standard digital zoom or resizing techniques often lead to blurry, pixelated, and unappealing images. This implementation uses a Generative Adversarial Network (GAN) to intelligently upscale images by "imagining" and generating the missing high-frequency details.

The core of the project consists of two competing neural networks:

* **A Generator Network:** A deep residual network responsible for upscaling the low-resolution input image.
* **A Discriminator Network:** A convolutional network trained to distinguish between real high-resolution images and the "fake" upscaled images created by the generator.

Through this adversarial training process, the generator becomes exceptionally skilled at creating sharp, detailed, and realistic high-resolution images.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

You will need Python 3.8+ and the following libraries:

* PyTorch
* NumPy
* Matplotlib
* scikit-image
* Pillow

### Installation

1. **Clone the repository:**
   ```
   git clone [https://github.com/Aditya-8807/srgan.git](https://github.com/Aditya-8807/srgan.git)
   cd srgan
   ```

2. **Install the required packages:**
   It's recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the Dataset:**
   The training script is configured to use the DIV2K dataset. You will need to download the training images and place them in the appropriate directories i.e srgan/data/.

## Usage

The project is structured for a simple and modular workflow.

### 1. Training the Model

To start training the model from scratch or resume from the last checkpoint, run the following command from the project's root directory:
```
python -m src.train
```
The script will automatically save versioned checkpoints at the epochs specified in `src/config.py`. It also saves the full training history for resuming and plotting.


### 3. Generating New Images

To use your trained model to upscale new images:

1. Place your low-resolution images into the `./test_images/lr/` directory.
2. In `src/test_on_new_images.py`, specify the epoch number of the trained model you wish to use.
3. Run the generation script:
   ```
   python -m src.test_on_new_images
   ```
The new high-resolution images will be saved in the `./test_images/sr/` directory.

## Project Structure

The project follows a clean and organized structure:

```
srgan/
├── benchmark/           # Benchmark datasets (e.g., Set5)
├── checkpoints/         # Saved model checkpoints (.pth) and history (.npz)
├── results/
│   ├── generated_images/  # Sample images saved during training
│   └── logs/            # Saved performance graphs
├── test_images/
│   ├── lr/              # Folder for low-res images to be tested
│   └── sr/              # Folder for generated high-res 
images
├──srgan.ipynb           #Google Colab notebook where model is trained 
└── src/
    ├── models/
    │   ├── generator.py
    │   └── discriminator.py
    ├── utils/
    │   ├── data_loader.py
    │   ├── losses.py
    │   └── metrics.py
    ├── config.py          # All project configurations
    ├── train.py           # Main training script
    ├── evaluate.py        # Evaluation script
    ├── plot_results.py    # Script to plot graphs
    └── test_on_new_images.py # Script for generating new images
```

## Results

After training, the model demonstrates a significant improvement in image quality, successfully generating fine details and textures.
