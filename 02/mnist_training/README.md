# MNIST Training Assignment

This project explores the use of different optimizers, learning rates, and decay schedules to train a neural network on the MNIST dataset. The assignment focuses on experimenting with these parameters to understand their impact on training performance.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The objective of this assignment is to implement and train a model on the MNIST dataset using:
1. **Optimizers**: Supports both `SGD` (with optional momentum) and `Adam`.
2. **Learning Rate and Schedules**:
    - A specified initial learning rate.
    - Optional learning rate decay schedules: `linear`, `exponential`, or `cosine`, each gradually reducing the learning rate to a specified final value.

**Example Commands**:
```bash
python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01
```

For more detailed usage examples, refer to the [Assignment Examples](#usage).

## Solution Explanation

The solution (`mnist_training.py`) includes:
1. **Model Construction**:
    - A simple neural network with a hidden layer of configurable size, followed by a softmax output layer for digit classification.
2. **Configurable Parameters**:
    - **Optimizer**: Select between `SGD` and `Adam`.
    - **Learning Rate Decay**: Decay schedules are implemented using Keras decay schedules:
        - **Linear Decay**: Decreases linearly from the initial to the final learning rate.
        - **Exponential Decay**: Decays exponentially based on the total training steps.
        - **Cosine Decay**: Applies a cosine-based decay over time.
3. **Training and Logging**:
    - Uses TensorBoard for logging metrics during training.
    - Callback logs training and validation metrics at each epoch for monitoring.

## Usage

To train the model with various configurations, use the following commands:

### Basic Training
```bash
python3 mnist_training.py --epochs=<number_of_epochs> --optimizer=<optimizer> --learning_rate=<learning_rate>
```

Example:
```bash
python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01
```

### Training with Momentum (for SGD only)
```bash
python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01 --momentum=0.9
```

### Training with Learning Rate Decay
```bash
python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001
```

For more usage examples, please refer to the example commands provided in the [Assignment Information](#assignment-overview).

## Files Structure

The directory structure for this project:

```
.
├── img                          # Images for visualizations.
├── logs                         # Directory for TensorBoard logs.
├── loss_landscape.ipynb         # Notebook for visualizing loss landscapes.
├── loss_values.npy              # Precomputed loss values for visualization.
├── mnist.npz                    # MNIST dataset file.
├── mnist.py                     # Utility for loading MNIST data.
├── mnist_training.py            # Main script for training and evaluation.
├── model.weights.h5             # Saved model weights.
├── neuron_visualization.ipynb   # Notebook for visualizing neuron activations.
└── README.md                    # Project documentation.
```

For details on each file, refer to the code documentation or comments in the respective files.