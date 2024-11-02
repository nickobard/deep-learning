# MNIST Ensemble Assignment

This project implements model ensembling on the MNIST dataset, with multiple models trained independently and combined to improve accuracy. The goal is to observe the accuracy gains from combining predictions across an increasing number of models.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The objective of this assignment is to:
1. **Train Multiple Models**:
    - Train a specified number (`args.models`) of individual models on the MNIST dataset.
2. **Evaluate Ensembling Performance**:
    - Evaluate each model individually on the development set.
    - Ensemble models incrementally (first model, first two models, etc.) by averaging their predictions, and evaluate the ensemble accuracy on the development set.

**Example Command**:
```bash
python3 mnist_ensemble.py --epochs=1 --models=5
```

## Solution Explanation

The solution (`mnist_ensemble.py`) includes:
1. **Model Training**:
    - The script creates multiple neural networks with a configurable number of hidden layers and units.
    - Each model is trained individually using the Adam optimizer and cross-entropy loss.
2. **Accuracy Evaluation**:
    - For each model, the script calculates individual accuracy on the development set.
    - Ensemble accuracy is calculated by averaging predictions from multiple models and evaluating the combined predictions on the development set.
3. **Incremental Ensembling**:
    - Starting from the first model, each additional model is added to the ensemble, and the new ensemble accuracy is evaluated. This provides insight into how accuracy improves as more models are combined.

## Usage

To run the ensemble training and evaluation with different parameters, use the following commands:

### Basic Ensembling with Default Layers
```bash
python3 mnist_ensemble.py --epochs=<number_of_epochs> --models=<number_of_models>
```

Example:
```bash
python3 mnist_ensemble.py --epochs=1 --models=5
```

### Configuring Hidden Layers
```bash
python3 mnist_ensemble.py --epochs=1 --models=5 --hidden_layers 100 200
```

### Example Output
```
Model 1, individual accuracy 96.04, ensemble accuracy 96.04
Model 2, individual accuracy 96.28, ensemble accuracy 96.56
Model 3, individual accuracy 96.12, ensemble accuracy 96.58
Model 4, individual accuracy 95.92, ensemble accuracy 96.70
Model 5, individual accuracy 96.38, ensemble accuracy 96.72
```

For more examples, refer to the example commands provided in the [Assignment Information](#assignment-overview).

## File Structure

The project directory contains the following files:

```
.
├── mnist.py              # Utility for loading MNIST data.
├── mnist_ensemble.py     # Main script for training and evaluating model ensembles.
└── README.md             # Project documentation.
```

Refer to individual file comments and code documentation for further details.