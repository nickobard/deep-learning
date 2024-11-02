# MNIST Regularization Assignment

This project implements three regularization techniques—dropout, weight decay, and label smoothing—to improve model generalization on the MNIST dataset. By configuring these methods, the assignment explores their effects on training and validation accuracy.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The objective of this assignment is to implement the following regularization techniques:
1. **Dropout**:
    - Adds dropout layers after the initial `Flatten` layer and each hidden `Dense` layer, with dropout rate specified by `args.dropout`.
2. **AdamW with Weight Decay**:
    - Uses the AdamW optimizer with weight decay specified by `args.weight_decay`, excluding biases from the decay.
3. **Label Smoothing**:
    - Implements label smoothing using the `args.label_smoothing` parameter to smooth class labels during cross-entropy loss calculation.

In addition to submitting the solution, it’s important to experiment with different values for each regularization method and observe the results in TensorBoard.

**Example Parameters to Experiment With**:
- **Dropout Rate**: 0, 0.3, 0.5, 0.6, 0.8
- **Weight Decay**: 0, 0.1, 0.3, 0.5, 1.0
- **Label Smoothing**: 0, 0.1, 0.3, 0.5

**Example Command**:
```bash
python3 mnist_regularization.py --epochs=1 --dropout=0.3
```

## Solution Explanation

The solution (`mnist_regularization.py`) includes:
1. **Model Architecture**:
    - A neural network with an initial `Flatten` layer, followed by configurable hidden layers with dropout applied after each.
    - A softmax output layer without dropout.
2. **Regularization Techniques**:
    - **Dropout**: Configured via `args.dropout`, it applies dropout to specified layers to reduce overfitting.
    - **AdamW with Weight Decay**: Configured via `args.weight_decay`, it uses the AdamW optimizer, ensuring that biases are excluded from weight decay.
    - **Label Smoothing**: Configured via `args.label_smoothing`, it utilizes categorical cross-entropy with smoothed labels.
3. **TensorBoard Logging**:
    - Logs training, development, and test set metrics for accuracy and loss, which can be visualized in TensorBoard for comparative analysis.

## Usage

To train the model with different regularization settings, use the following commands:

### Dropout Example
```bash
python3 mnist_regularization.py --epochs=1 --dropout=0.3
```

### Weight Decay Example
```bash
python3 mnist_regularization.py --epochs=1 --weight_decay=0.1
```

### Label Smoothing Example
```bash
python3 mnist_regularization.py --epochs=1 --label_smoothing=0.1
```

### Combined Regularization Example
```bash
python3 mnist_regularization.py --epochs=1 --dropout=0.5 --weight_decay=0.1 --label_smoothing=0.2
```

## File Structure

The project directory structure is as follows:

```
.
├── logs                       # Directory for TensorBoard logs.
├── mnist.npz                  # MNIST dataset file.
├── mnist.py                   # Utility script for loading and batching MNIST data.
├── mnist_regularization.py    # Main script for training with regularization.
└── README.md                  # Project documentation.
```

For detailed explanations and results, refer to individual comments within the code and review logs in TensorBoard.