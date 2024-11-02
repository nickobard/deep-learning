## Table of Contents
- [Projects Overview](#projects-overview)
    - [Gym CartPole](#gym-cartpole)
    - [MNIST Training](#mnist-training)
    - [SGD Backpropagation](#sgd-backpropagation)
    - [SGD Manual Gradient Computation](#sgd-manual-gradient-computation)
- [File Structure](#file-structure)

---

## Projects Overview

### Gym CartPole

The **Gym CartPole** project aims to solve the CartPole-v1 environment from the Gymnasium library using supervised learning with a limited dataset. The model is trained to predict actions based on input observations to achieve a high score on the CartPole task.

- **Key Features**:
    - Predicts actions using a neural network model with either sigmoid or softmax output.
    - Trains with optimizers like SGD or Adam and optional learning rate decay.
    - Achieves a high average score in the CartPole environment.

**Example Command**:
```bash
python3 gym_cartpole.py --evaluate --render
```

For details, refer to the [Gym CartPole README](gym_cartpole/README.md).

---

### MNIST Training

The **MNIST Training** project explores training a neural network on the MNIST dataset using various optimizers, learning rates, and decay schedules.

- **Key Features**:
    - Supports `SGD` (with optional momentum) and `Adam` optimizers.
    - Implements different learning rate decay schedules: linear, exponential, and cosine.
    - Utilizes TensorBoard for visualizing training and validation metrics.

**Example Command**:
```bash
python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01
```

For detailed usage, see the [MNIST Training README](mnist_training/README.md).

---

### SGD Backpropagation

The **SGD Backpropagation** project focuses on implementing minibatch SGD with backpropagation on the MNIST dataset. This exercise is designed to familiarize with PyTorch's automatic differentiation.

- **Key Features**:
    - Implements a neural network with a single hidden layer (tanh activation) and a softmax output.
    - Uses `.backward()` for gradient computation.
    - Updates weights using manually implemented SGD.

**Example Command**:
```bash
python3 sgd_backpropagation.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1
```

For more details, refer to the [SGD Backpropagation README](sgd_backpropagation/README.md).

---

### SGD Manual Gradient Computation

The **SGD Manual Gradient Computation** project extends the previous backpropagation assignment by manually computing gradients instead of using automatic differentiation. This exercise provides a deeper understanding of the mathematics of backpropagation.

- **Key Features**:
    - Computes gradients manually for each layer and parameter.
    - Updates model weights with manually calculated gradients in minibatch SGD.
    - Evaluates model performance after each epoch.

**Example Command**:
```bash
python3 sgd_manual.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1
```

For a complete explanation, refer to the [SGD Manual Gradient Computation README](sgd_manual/README.md).

---

## Files Structure

The directory structure for this repository:

```
.
├── gym_cartpole
│   ├── .gitignore
│   ├── gym_cartpole_data.txt
│   ├── gym_cartpole.py
│   ├── logs
│   └── README.md
├── mnist_training
│   ├── img
│   │   ├── d1_dot_d2.png
│   │   ├── gramm_schmidt_1.png
│   │   └── gramm_schmidt_2.png
│   ├── logs
│   ├── loss_landscape.ipynb
│   ├── loss_values.npy
│   ├── mnist.npz
│   ├── mnist.py
│   ├── mnist_training.py
│   ├── model.weights.h5
│   ├── neuron_visualization.ipynb
│   └── README.md
├── sgd_backpropagation
│   ├── mnist.py
│   ├── README.md
│   └── sgd_backpropagation.py
└── sgd_manual
    ├── mnist.py
    ├── README.md
    └── sgd_manual.py
```

Each project contains a dedicated `README.md` file with specific usage instructions and a detailed description of its purpose and functionality. Explore individual project directories for more information.