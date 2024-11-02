## Table of Contents
- [Projects Overview](#projects-overview)
    - [Uppercase Conversion](#uppercase-conversion)
    - [MNIST Regularization](#mnist-regularization)
    - [MNIST Ensemble](#mnist-ensemble)
- [File Structure](#file-structure)

---

## Projects Overview

### Uppercase Conversion

The **Uppercase Conversion** project is an NLP task focused on converting lowercased Czech text to the correct uppercase form. Using a fully connected neural network, this project applies sliding window techniques to predict the appropriate capitalization for each character.

- **Key Features**:
    - Predicts uppercase letters using a sliding window approach for context.
    - Utilizes TensorBoard logging for training progress.
    - Evaluates predictions based on character accuracy.

**Example Command**:
```bash
python3 uppercase.py --epochs=5 --window=4 --alphabet_size=100
```

For detailed instructions, refer to the [Uppercase Conversion README](uppercase/README.md).

---

### MNIST Regularization

The **MNIST Regularization** project enhances model performance and generalization using regularization techniques such as dropout, weight decay (AdamW), and label smoothing. This project demonstrates the impact of these techniques on training and validation accuracy.

- **Key Features**:
    - Dropout layers applied after `Flatten` and each hidden `Dense` layer.
    - AdamW optimizer with configurable weight decay.
    - Label smoothing for better classification margin and accuracy.

**Example Command**:
```bash
python3 mnist_regularization.py --epochs=1 --dropout=0.3
```

For more details, see the [MNIST Regularization README](mnist_regularization/README.md).

---

### MNIST Ensemble

The **MNIST Ensemble** project implements model ensembling on the MNIST dataset, training multiple models independently and combining their predictions to boost accuracy. The project showcases the incremental accuracy gains from adding more models to the ensemble.

- **Key Features**:
    - Trains multiple independent models on the MNIST dataset.
    - Calculates ensemble accuracy by averaging predictions from increasing subsets of models.
    - Monitors the improvement in accuracy as more models are added to the ensemble.

**Example Command**:
```bash
python3 mnist_ensemble.py --epochs=1 --models=5
```

Refer to the [MNIST Ensemble README](mnist_ensemble/README.md) for more information.

---

## File Structure

The directory structure for this repository:

```
.
├── mnist_ensemble
│   ├── mnist.py
│   ├── mnist_ensemble.py
│   └── README.md
├── mnist_regularization
│   ├── logs
│   ├── mnist.npz
│   ├── mnist.py
│   ├── mnist_regularization.py
│   └── README.md
├── uppercase
│   ├── logs
│   ├── README.md
│   ├── uppercase_data.py
│   └── uppercase.py
└── README.md
```

Each project contains its own `README.md` file with specific usage instructions, solution explanations, and file structure details.