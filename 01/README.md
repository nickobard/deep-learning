## Table of Contents
- [Project Overview](#project-overview)
- [Projects](#projects)
    - [Numpy Entropy](#numpy-entropy)
    - [PCA First](#pca-first)
    - [MNIST Layers Activations](#mnist-layers-activations)
- [File Structure](#file-structure)

---

## Project Overview

Each project focuses on a specific machine learning concept:
1. **Numpy Entropy**: Calculates entropy, cross-entropy, and KL-divergence using probability distributions.
2. **PCA First**: Implements principal component analysis on the MNIST dataset using PyTorch tensors.
3. **MNIST Layers Activations**: Builds a configurable neural network with user-specified hidden layers and activation functions, including TensorBoard visualization.

## Projects

### [Numpy Entropy](numpy_entropy/README.md)

The **Numpy Entropy** project is an introduction to data science metrics using NumPy, implementing:
- **Entropy**: Quantifies the uncertainty in a probability distribution.
- **Cross-entropy**: Measures the dissimilarity between two distributions.
- **KL-divergence**: Calculates how one probability distribution diverges from a second expected probability distribution.

**Usage**:
```bash
python3 numpy_entropy.py --data_path <data_file> --model_path <model_file>
```

For detailed information, refer to the [Numpy Entropy README](numpy_entropy/README.md).

---

### [PCA First](pca_first/README.md)

The **PCA First** project demonstrates principal component analysis (PCA) on the MNIST dataset:
- **Covariance Calculation**: Computes the covariance matrix of the data.
- **Power Iteration**: Finds the first principal component.
- **Explained Variance**: Calculates the variance explained by the first principal component.

**Usage**:
```bash
python3 pca_first.py --examples=<number_of_examples> --iterations=<number_of_iterations>
```

For more information, see the [PCA First README](pca_first/README.md).

---

### [MNIST Layers Activations](mnist_layers_activations/README.md)

The **MNIST Layers Activations** project involves building and training a customizable neural network on the MNIST dataset using Keras:
- **Configurable Layers**: Specifies the number and type of hidden layers using command-line parameters.
- **Activation Functions**: Supports various activation functions (`none`, `relu`, `tanh`, `sigmoid`) for each layer.
- **TensorBoard Integration**: Logs training progress for visualization in TensorBoard.

**Usage**:
```bash
python3 mnist_layers_activations.py --epochs=<number_of_epochs> --hidden_layers=<number_of_layers> --activation=<activation_function>
```

For detailed instructions, refer to the [MNIST Layers Activations README](mnist_layers_activations/README.md).

---

## File Structure

```
.
├── mnist_layers_activations
│   ├── example_keras.py
│   ├── example_keras_tensorboard.py
│   ├── mnist_layers_activations.py
│   ├── mnist.py
│   └── README.md
├── numpy_entropy
│   ├── numpy_entropy_data_1.txt
│   ├── numpy_entropy_data_2.txt
│   ├── numpy_entropy_data_3.txt
│   ├── numpy_entropy_data_4.txt
│   ├── numpy_entropy_model_1.txt
│   ├── numpy_entropy_model_2.txt
│   ├── numpy_entropy_model_3.txt
│   ├── numpy_entropy_model_4.txt
│   ├── numpy_entropy.py
│   └── README.md
├── pca_first
│   ├── mnist.py
│   ├── pca_first.py
│   └── README.md
└── README.md
```

Each subdirectory contains its own `README.md` with additional details. Explore individual READMEs for specific implementation details and usage examples.