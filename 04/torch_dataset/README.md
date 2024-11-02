# CIFAR-10 Dataset Augmentation Assignment

This project explores dataset augmentation techniques using PyTorch’s `torch.utils.data` module for constructing training datasets. The assignment uses the CIFAR-10 dataset to apply various image transformations, enhancing model robustness during training.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The goal of this assignment is to:
1. **Familiarize with PyTorch’s `torch.utils.data`**:
    - Implement dataset handling, augmentation, and loading with PyTorch.
2. **Apply Data Augmentation**:
    - Use image transformations like resizing, padding, cropping, and horizontal flipping to increase dataset variety and improve model generalization.
3. **Train a Convolutional Neural Network on CIFAR-10**:
    - Utilize a simple CNN model to classify CIFAR-10 images, with augmentation applied optionally.

**Supported Augmentations**:
- Random resizing (between 28x28 and 36x36 pixels).
- Padding and random cropping to 32x32 pixels.
- Random horizontal flipping.

**Example Command**:
```bash
python3 torch_dataset.py --epochs=1 --batch_size=100 --augment
```

## Solution Explanation

The solution (`torch_dataset.py`) includes:
1. **Dataset Setup**:
    - Implements the `TorchDataset` class, which uses the `torch.utils.data.Dataset` interface to load CIFAR-10 images and labels.
    - Includes an optional `augmentation_fn` for applying transformations to training data.
2. **Image Augmentation**:
    - If the `--augment` flag is set, a sequence of transformations (`v2.Compose`) is applied to each image, including random resizing, padding, cropping, and flipping.
3. **Data Loading and Training**:
    - The `DataLoader` class is used to create batches for both training and validation datasets.
    - A simple CNN model classifies CIFAR-10 images with the following structure:
        - Multiple convolutional and dense layers, with ReLU activations and softmax output.
4. **TensorBoard Visualization**:
    - If the `--show_images` argument is provided, the script logs a grid of transformed images to TensorBoard for visualization.

## Usage

To train the model with or without augmentation, use the following command:

### Basic Training Command
```bash
python3 torch_dataset.py --epochs=<number_of_epochs> --batch_size=<batch_size> [--augment] [--show_images=<grid_size>]
```

Example:
```bash
python3 torch_dataset.py --epochs=1 --batch_size=50 --augment
```

### Example Output
For `--epochs=1` and `--batch_size=50` with augmentation, example output:
```
accuracy: 0.1354 - loss: 2.2565 - val_accuracy: 0.2690 - val_loss: 1.9889
```

## File Structure

The project directory contains the following files:

```
.
├── cifar10.py                # CIFAR-10 dataset utility.
├── torch_dataset.py          # Main script for dataset handling, augmentation, and training.
├── logs                      # Directory for TensorBoard logs.
│   └── augmentations
└── README.md                 # Project documentation.
```

Refer to individual file comments and code documentation for further details on augmentation and model training.