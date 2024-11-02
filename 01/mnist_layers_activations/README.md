# MNIST Layers Activations Assignment

This project explores neural network construction in Keras, with a focus on configuring hidden layers and activation functions for a model trained on the MNIST dataset. The project also demonstrates using TensorBoard to monitor model performance and metrics.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [TensorBoard Usage](#tensorboard-usage)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

In this assignment:
1. **Setup and Familiarize with TensorBoard**: Run `example_keras_tensorboard.py` to explore TensorBoard’s functionalities. Use `tensorboard --logdir logs` and open `http://localhost:6006` to inspect metrics.
2. **Configure Model Layers and Activations**:
    - A specified number of hidden layers (including zero) can be added, defined by the `--hidden_layers` parameter.
    - Each layer’s activation function is set through the `--activation` parameter, supporting `none`, `relu`, `tanh`, and `sigmoid`.
3. **Train and Evaluate Model**:
    - Train the model on the MNIST dataset, outputting metrics like accuracy and loss.
    - Use TensorBoard to monitor metrics for each epoch.

**Example Commands**:
```bash
python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=relu
```

**Expected Output Format**:
```plaintext
accuracy: 0.8503 - loss: 0.5286 - val_accuracy: 0.9604 - val_loss: 0.1432
```

## Solution Explanation

The solution (`mnist_layers_activations.py`) includes:
1. **Argument Parsing**: Configurable command-line parameters include `--hidden_layers` and `--activation`.
2. **Dynamic Model Construction**:
    - The model begins with input and normalization layers, followed by a specified number of hidden layers.
    - Each hidden layer’s activation function is applied as specified by the `--activation` argument.
    - Finally, a softmax output layer classifies the MNIST digits.
3. **TensorBoard Integration**:
    - Logs are saved for visualization in TensorBoard, providing insights into training/validation loss and accuracy.
    - The callback, `TorchTensorBoardCallback`, saves logs per epoch for easy monitoring.

## TensorBoard Usage

To launch TensorBoard and explore model training logs:
1. Run:
   ```bash
   tensorboard --logdir logs
   ```
2. Open your browser and go to `http://localhost:6006`.

## Usage

Run the script with specific parameters as needed:
```bash
python3 mnist_layers_activations.py --epochs=<number_of_epochs> --hidden_layers=<number_of_layers> --activation=<activation_function>
```

Example:
```bash
python3 mnist_layers_activations.py --epochs=10 --hidden_layers=3 --activation=relu
```

## File Structure

The project files include:
- `mnist_layers_activations.py`: Main script for configuring and training the MNIST model with specified layers and activations.
- `example_keras.py`: A simple example demonstrating basic Keras model setup.
- `example_keras_tensorboard.py`: Example showcasing TensorBoard logging with Keras.
- `mnist.py`: Utility module for loading and processing MNIST data.
- `README.md`: Project documentation.
