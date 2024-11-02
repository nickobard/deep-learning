# SGD Backpropagation Assignment

This project implements a neural network trained on the MNIST dataset using a manually implemented stochastic gradient descent (SGD) algorithm with backpropagation for gradient calculation. The objective is to familiarize with PyTorch's automatic differentiation and minibatch SGD.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The goal of this assignment is to:
1. **Implement a Neural Network**:
    - A single hidden layer with tanh activation and a softmax output layer.
2. **Compute Cross-Entropy Loss**:
    - Calculate cross-entropy loss over a minibatch.
3. **Use Autograd for Gradient Calculation**:
    - Employ `.backward()` to compute gradients with respect to all model variables.
4. **Perform SGD Updates**:
    - Update model weights using manually implemented minibatch SGD with a specified learning rate.

**Example Commands**:
```bash
python3 sgd_backpropagation.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1
```

For detailed examples, refer to the [Assignment Examples](#usage).

## Solution Explanation

The solution (`sgd_backpropagation.py`) includes:
1. **Model Architecture**:
    - A neural network with one hidden layer (tanh activation) and a categorical softmax output layer.
2. **Training with Backpropagation**:
    - Each epoch iterates through minibatches, using `torch.autograd` to calculate gradients.
    - Cross-entropy loss is computed for each batch, followed by gradient backpropagation with `.backward()`.
3. **SGD Update**:
    - Weights are updated using SGD, applying gradients scaled by the learning rate to each parameter.
4. **Evaluation**:
    - After each epoch, the model is evaluated on the development set.
    - The final accuracy is computed on both the development and test sets.

## Usage

To train the model with different configurations, use the following commands:

### Basic Training
```bash
python3 sgd_backpropagation.py --epochs=<number_of_epochs> --batch_size=<batch_size> --hidden_layer=<hidden_layer_size> --learning_rate=<learning_rate>
```

Example:
```bash
python3 sgd_backpropagation.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1
```

### Example Output
```
Dev accuracy after epoch 1 is 93.30
Dev accuracy after epoch 2 is 94.38
Test accuracy after epoch 2 is 93.15
```

For more usage examples, please refer to the example commands provided in the [Assignment Information](#assignment-overview).

## Files Structure

The directory structure for this project:

```
.
├── mnist.py                   # Utility for loading and batching MNIST data.
├── sgd_backpropagation.py     # Main script for training and evaluation.
└── README.md                  # Project documentation.
```

For details on each file, refer to the code documentation or comments within the respective files.