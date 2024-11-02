## Table of Contents
- [Assignments Overview](#assignments-overview)
    - [MNIST CNN](#mnist-cnn)
    - [MNIST Multiple Inputs and Outputs](#mnist-multiple-inputs-and-outputs)
    - [CIFAR-10 Dataset Augmentation](#cifar-10-dataset-augmentation)
- [File Structure](#file-structure)

---

## Assignments Overview

### MNIST CNN

This assignment explores a configurable CNN architecture for the MNIST dataset. Various layer types and configurations are defined through command-line arguments, supporting convolutional layers, batch normalization, residual connections, and dropout.

- **Key Features**:
    - Construct CNNs using different layer configurations via command-line arguments.
    - Train and evaluate the CNN on MNIST, monitoring accuracy and loss.

**Example Command**:
```bash
python3 mnist_cnn.py --epochs=10 --cnn="CB-16-5-2-same,M-3-2,F,H-100,D-0.5"
```

For more information, see the [MNIST CNN README](mnist_cnn/README.md)【137†source】.

---

### MNIST Multiple Inputs and Outputs

This project implements a multi-input, multi-output model on MNIST, where two MNIST images are inputted, and the model predicts if the digit in the first image is greater than that in the second. The model has four outputs, supporting both direct and indirect comparisons between the images.

- **Key Features**:
    - Multi-input and multi-output model with both direct and indirect comparison outputs.
    - Dataset creation function to construct pairs of images with comparison labels.

**Example Command**:
```bash
python3 mnist_multiple.py --epochs=1 --batch_size=50
```

For more details, refer to the [MNIST Multiple README](mnist_multiple/README.md)【138†source】.

---

### CIFAR-10 Dataset Augmentation

In this assignment, various image augmentation techniques are applied to the CIFAR-10 dataset using PyTorch’s `torch.utils.data` module. The augmentations aim to improve model generalization by introducing variety in the training data.

- **Key Features**:
    - Augmentations include resizing, padding, cropping, and flipping.
    - Optionally log transformed images to TensorBoard.

**Example Command**:
```bash
python3 torch_dataset.py --epochs=1 --batch_size=50 --augment
```

For further information, see the [CIFAR-10 Dataset Augmentation README](torch_dataset/README.md)【139†source】.

---

## File Structure

The directory structure for this repository:

```
.
├── cifar_competition
│   ├── cifar10.py
│   └── cifar_competition.py
├── mnist_cnn
│   ├── mnist.py
│   ├── mnist_cnn.py
│   └── README.md
├── mnist_multiple
│   ├── figures
│   │   ├── mnist_multiple.drawio
│   │   ├── mnist_multiple.png
│   │   └── mnist_multiple.svgz
│   ├── mnist.py
│   ├── mnist_multiple.py
│   └── README.md
├── torch_dataset
│   ├── cifar10.py
│   ├── logs
│   │   └── augmentations
│   ├── README.md
│   └── torch_dataset.py
└── README.md
```

Each project contains its own `README.md` file with specific usage instructions, solution explanations, and detailed file structure descriptions.