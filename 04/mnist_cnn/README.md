# MNIST CNN Assignment

This project implements a configurable convolutional neural network (CNN) for the MNIST dataset, supporting various layer types and architectures defined via a command-line argument. The goal is to explore and evaluate CNN configurations, including convolutional layers, batch normalization, residual connections, and dropout.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The objective of this assignment is to:
1. **Construct a CNN Architecture**:
    - Define and parse a CNN configuration string passed as a command-line argument.
    - Supported layers include convolution, batch normalization, max pooling, residual connections, flattening, dense layers, and dropout.
2. **Train and Evaluate the CNN**:
    - Train the configured CNN on the MNIST dataset for a specified number of epochs.
    - Evaluate model performance on the development set, recording accuracy and loss.

**Supported Layer Types**:
- **Convolution (`C` and `CB`)**: Adds a convolutional layer, optionally with batch normalization (`CB`).
- **Max Pooling (`M`)**: Adds a max pooling layer.
- **Residual Block (`R`)**: Adds a residual block containing one or more convolutional layers.
- **Flatten (`F`)**: Flattens the input, required once in the architecture.
- **Dense Layer (`H`)**: Adds a fully connected layer.
- **Dropout (`D`)**: Applies dropout with a specified rate.

**Example Configurations**:
```bash
--cnn="CB-16-5-2-same,M-3-2,F,H-100,D-0.5"
```

For detailed architecture specifications, refer to the layer descriptions in the code or in this README’s Solution Explanation.

## Solution Explanation

The solution (`mnist_cnn.py`) includes:
1. **Model Architecture Parsing**:
    - The script uses regular expressions to parse the `--cnn` argument, creating a sequence of layers based on the specifications provided.
2. **Supported Layers**:
    - **Convolution (`C`)**: Adds a convolutional layer with ReLU activation.
    - **Convolution with Batch Norm (`CB`)**: Adds a convolutional layer followed by batch normalization and ReLU activation.
    - **Max Pooling (`M`)**: Adds a max pooling layer with the specified pool size and stride.
    - **Residual Block (`R`)**: Creates a residual connection that sums the input with the output of specified convolutional layers.
    - **Flatten (`F`)**: Flattens the input, required exactly once.
    - **Dense Layer (`H`)**: Adds a dense (fully connected) layer with ReLU activation.
    - **Dropout (`D`)**: Adds a dropout layer with the specified dropout rate.
3. **Model Training**:
    - The model is trained using the Adam optimizer with sparse categorical cross-entropy loss.
    - Logs are maintained for accuracy and loss metrics, which can be viewed for performance evaluation.

## Usage

To train the model with a specific architecture, use the following command:

### Basic Training Command
```bash
python3 mnist_cnn.py --epochs=<number_of_epochs> --cnn="<cnn_configuration>"
```

Example:
```bash
python3 mnist_cnn.py --epochs=10 --cnn="CB-16-5-2-same,M-3-2,F,H-100,D-0.5"
```

### Example Outputs
For the configuration `--cnn="CB-16-5-2-same,M-3-2,F,H-100,D-0.5"`, an example output would be:
```
Epoch 1/10 accuracy: 0.7706 - loss: 0.7444 - val_accuracy: 0.9572 - val_loss: 0.1606
Epoch 2/10 accuracy: 0.9177 - loss: 0.2808 - val_accuracy: 0.9646 - val_loss: 0.1286
...
Epoch 10/10 accuracy: 0.9560 - loss: 0.1413 - val_accuracy: 0.9754 - val_loss: 0.0792
```

## File Structure

The project directory contains the following files:

```
.
├── mnist_cnn.py            # Main script for training and evaluating CNN on MNIST.
├── mnist.py                # Utility for loading and batching MNIST data.
└── README.md               # Project documentation.
```

Refer to individual file comments and code documentation for additional details.