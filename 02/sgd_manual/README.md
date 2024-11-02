# SGD Manual Gradient Computation Assignment

This project implements a neural network trained on the MNIST dataset, performing minibatch stochastic gradient descent (SGD) with manually computed gradients. This exercise deepens understanding of the mathematics behind backpropagation by manually calculating gradients instead of relying on automatic differentiation.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The goal of this assignment is to:
1. **Implement a Neural Network**:
    - A single hidden layer with tanh activation, followed by a softmax output layer.
2. **Compute Cross-Entropy Loss**:
    - Calculate cross-entropy loss over a minibatch.
3. **Manually Compute Gradients**:
    - Derive and compute gradients for all parameters manually, ensuring accurate updates during training.
4. **Perform SGD Updates**:
    - Use computed gradients to update model weights with minibatch SGD.

**Example Commands**:
```bash
python3 sgd_manual.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1
```

## Solution Explanation

The solution (`sgd_manual.py`) includes:
1. **Model Architecture**:
    - A neural network with one hidden layer (tanh activation) and a softmax output layer.
2. **Manual Gradient Calculation**:
    - During training, forward and backward passes are completed manually, with each parameter’s gradient derived and computed without using automatic differentiation.
    - For each batch:
        - The forward pass computes predictions and intermediate activations.
        - The backward pass calculates gradients for each layer by hand, including those for weights and biases.
3. **SGD Update**:
    - Model weights and biases are updated using manually computed gradients scaled by the learning rate.
4. **Evaluation**:
    - After each epoch, the model is evaluated on the development set.
    - Final accuracy is computed on both development and test sets.

## Usage

To train the model with different configurations, use the following commands:

### Basic Training
```bash
python3 sgd_manual.py --epochs=<number_of_epochs> --batch_size=<batch_size> --hidden_layer=<hidden_layer_size> --learning_rate=<learning_rate>
```

Example:
```bash
python3 sgd_manual.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1
```

### Example Output
```
Dev accuracy after epoch 1 is 93.30
Dev accuracy after epoch 2 is 94.38
Test accuracy after epoch 2 is 93.15
```

For more usage examples, refer to the example commands provided in the [Assignment Information](#assignment-overview).

## Files Structure

The directory structure for this project:

```
.
├── mnist.py                   # Utility for loading and batching MNIST data.
├── sgd_manual.py              # Main script for training with manual gradient computation.
└── README.md                  # Project documentation.
```

For details on each file, refer to the code documentation or comments within the respective files.