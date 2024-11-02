# Uppercase Conversion Task

This project addresses an NLP task in Czech text processing, where the objective is to convert lowercased text to the correct uppercase form. Using fully connected layers and techniques like sliding windows, this project evaluates different letter-casing patterns to achieve high accuracy in uppercase prediction.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The objective of this assignment is to:
1. **Train a Model to Predict Uppercase Characters**:
    - Given a lowercased text input, predict the appropriate capitalization for each character.
2. **Dataset and Evaluation**:
    - The dataset consists of training and development sets with correct casing, while the test set is fully lowercased.
    - Accuracy is computed as the percentage of correctly predicted characters.
3. **Submission Requirements**:
    - Submit a `.txt` file with the uppercased test set and a Python script (`.py`) or notebook (`.ipynb`) describing the approach.

**Restrictions**:
- The solution should not use RNNs, CNNs, or Transformers but may include fully connected layers, embeddings, activations, residual connections, and regularization.

**Example Command to Evaluate Accuracy**:
```bash
python3 uppercase_data.py --evaluate <predictions_file>
```

## Solution Explanation

The solution (`uppercase.py`) includes:
1. **Data Processing**:
    - The `UppercaseData` class loads the dataset, applies a sliding window approach to capture context around each character, and maps characters to indices in a specified alphabet.
2. **Model Architecture**:
    - The model is composed of:
        - An input layer with a specified window size.
        - One-hot encoding for character indices.
        - Dense hidden layers with a configurable activation function (`relu`, `tanh`, or `sigmoid`).
        - A sigmoid output layer for binary classification (uppercased or not).
    - The model uses the Adam optimizer with binary cross-entropy loss.
3. **Training and Evaluation**:
    - The model trains on the training set and validates on the development set.
    - TensorBoard logging is integrated for monitoring training progress.
4. **Prediction Generation**:
    - After training, predictions on the test set are generated and written to a `.txt` file, with uppercase applied to characters where the model’s probability exceeds 0.5.

## Usage

To train and generate predictions, use the following commands:

### Basic Training Command
```bash
python3 uppercase.py --epochs=<number_of_epochs> --window=<window_size> --alphabet_size=<alphabet_size>
```

Example:
```bash
python3 uppercase.py --epochs=5 --window=4 --alphabet_size=100
```

### Evaluating Predictions
To evaluate accuracy of the predictions on the development set:
```bash
python3 uppercase_data.py --evaluate <path_to_predictions_file>
```

## File Structure

The directory structure for this project:

```
.
├── logs                     # Directory for TensorBoard logs.
├── README.md                # Project documentation.
├── uppercase_data.py        # Data loading and evaluation utilities.
└── uppercase.py             # Main script for training and uppercase prediction.
```

For details on the implementation and specific parameters, review the comments within the code files.