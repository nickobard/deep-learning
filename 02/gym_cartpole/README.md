# Gym CartPole Assignment

This project aims to solve the CartPole-v1 environment from the Gymnasium library using supervised learning with limited training data. The task involves creating a model that can generalize well enough to achieve a high score on the CartPole task by predicting actions based on input observations.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The objective is to train a model to solve the CartPole-v1 environment using only a provided dataset (`gym_cartpole_data.txt`) of observations and actions:
1. **Training Data**: Each line in `gym_cartpole_data.txt` contains four floating-point numbers (observations) and one integer (the action taken for that observation).
2. **Model Requirements**:
    - The model must predict actions based on observations, using either one or two outputs (sigmoid or softmax activation).
    - To pass, the model should achieve an average reward of at least 475 over 100 episodes in evaluation mode.
3. **Evaluation**: The model is evaluated by running the CartPole-v1 environment with random initial states, aiming for an average score that meets or exceeds the threshold.

**Example Command for Evaluation**:
```bash
python3 gym_cartpole.py --evaluate --render
```

For more details on CartPole, visit the [CartPole-v1 environment documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

## Solution Explanation

The solution (`gym_cartpole.py`) includes:
1. **Model Architecture**:
    - A configurable neural network is built based on user-specified parameters for hidden layers, units, activation function, learning rate, and optimizer.
    - The output layer uses a sigmoid activation (for binary output) to predict actions based on input observations.
2. **Training**:
    - The model is trained on the provided dataset with a selected optimizer (SGD or Adam), and optional learning rate decay (linear, exponential, or cosine).
    - TensorBoard logging is integrated for training and validation metrics.
3. **Evaluation**:
    - During evaluation (`--evaluate`), the trained model is used to predict actions on randomly initialized episodes in the CartPole-v1 environment.
    - The `evaluate_model` function calculates the average score over the specified number of episodes, and if `--render` is set, the CartPole simulation will be displayed.

## Usage

To train or evaluate the model, use the following commands:

### Training
```bash
python3 gym_cartpole.py --epochs=<number_of_epochs> --hidden_layers=<number_of_layers> --activation=<activation_function>
```

Example:
```bash
python3 gym_cartpole.py --epochs=10 --hidden_layers=1 --activation=relu
```

### Evaluation
To evaluate a trained model:
```bash
python3 gym_cartpole.py --evaluate --render
```

## Files Structure

The project directory contains the following files:

```
.
├── gym_cartpole.py             # Main script for training and evaluating the CartPole model.
├── gym_cartpole_data.txt       # Training data with observations and actions.
├── logs                        # Directory for TensorBoard logs.
└── README.md                   # Project documentation.
```

Explore the individual files for implementation details, or refer to the documentation in this README for usage instructions.