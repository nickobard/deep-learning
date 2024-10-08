#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch
import torch.utils.tensorboard

from mnist import MNIST

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        self._W1 = keras.Variable(
            keras.random.normal([MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed),
            trainable=True,
        )
        self._b1 = keras.Variable(keras.ops.zeros([args.hidden_layer]), trainable=True)

        # Initialize trainable variables for the second layer
        # _W2: Weight matrix of shape [args.hidden_layer, MNIST.LABELS], initialized with a normal distribution
        #      (stddev=0.1) and using the specified seed for reproducibility
        self._W2 = keras.Variable(keras.random.normal([args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed),
                                  trainable=True)
        # _b2: Bias vector of shape [MNIST.LABELS], initialized to zeros
        self._b2 = keras.Variable(keras.ops.zeros([MNIST.LABELS]), trainable=True)

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through the network to compute predictions
        # Cast input images to float32 and normalize pixel values to [0, 1]
        casted_inputs = keras.ops.cast(inputs, dtype="float32")
        normalized = casted_inputs / 255
        # Flatten the images to shape [batch_size, num_features]
        prepared_inputs = normalized.reshape(inputs.shape[0], -1)
        # Compute the hidden layer pre-activation values
        hidden_layers = prepared_inputs @ self._W1 + self._b1
        # Apply tanh activation function
        activations = keras.ops.tanh(hidden_layers)
        # Compute the output layer logits
        outputs = activations @ self._W2 + self._b2
        # Apply softmax to obtain class probabilities
        predictions = keras.ops.softmax(outputs)

        # Return predictions and intermediate activations for manual gradient computation
        return predictions, activations, prepared_inputs

    def train_epoch(self, dataset: MNIST.Datasplit) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # Each batch contains:
            # - batch["images"] with shape [batch_size, MNIST.H, MNIST.W, MNIST.C]
            # - batch["labels"] with shape [batch_size]
            # The last batch may be smaller than `self._args.batch_size`.

            # Manually compute gradients without automatic differentiation
            # Perform forward pass to get predictions and intermediate activations
            inputs, labels = batch['images'], batch['labels']
            predictions, activations, prepared_inputs = self.predict(inputs)

            # Convert labels to one-hot encoding for loss computation
            labels = keras.ops.one_hot(labels, num_classes=MNIST.LABELS)

            # Compute gradients of the loss with respect to model parameters

            # Gradient of the loss with respect to the output logits (pre-softmax activations)
            z2_grad = predictions - labels
            # Gradient of the loss with respect to the second layer biases (_b2)
            b2_grad = keras.ops.mean(z2_grad, axis=0)

            # Gradient of the loss with respect to the second layer weights (_W2)
            # Calculated using batched outer product between activations and z2_grad
            W2_outer_prod = activations[:, :, np.newaxis] * z2_grad[:, np.newaxis, :]
            W2_grad = keras.ops.mean(W2_outer_prod, axis=0)

            # Compute derivative of the tanh activation function for the hidden layer
            a_grad = activations ** 2
            a_grad = 1 - a_grad

            # Gradient of the loss with respect to the hidden layer pre-activations (z1)
            W2 = self._W2
            z1_grad = W2 @ keras.ops.transpose(z2_grad)

            # Gradient of the loss with respect to the first layer biases (_b1)
            z1_grad = keras.ops.transpose(z1_grad) * a_grad
            b1_grad = keras.ops.mean(z1_grad, axis=0)

            # Gradient of the loss with respect to the first layer weights (_W1)
            # Calculated using batched outer product between inputs and z1_grad
            W1_outer_prod = prepared_inputs[:, :, np.newaxis] * z1_grad[:, np.newaxis, :]
            W1_grad = keras.ops.mean(W1_outer_prod, axis=0)

            # Update the model parameters using Stochastic Gradient Descent
            variables = [self._W1, self._b1, self._W2, self._b2]
            gradients = [W1_grad, b1_grad, W2_grad, b2_grad]
            for variable, gradient in zip(variables, gradients):
                # Update each variable by subtracting the learning rate times the gradient
                g_hat = gradient
                variable.assign_sub(g_hat * self._args.learning_rate)

    def evaluate(self, dataset: MNIST.Datasplit) -> float:
        # Compute the accuracy of the model predictions
        correct = 0
        for batch in dataset.batches(self._args.batch_size):
            inputs, labels = batch['images'], batch['labels']
            # Obtain predicted probabilities from the model and convert to a NumPy array
            probabilities = keras.ops.convert_to_numpy(self.predict(inputs)[0])

            # Determine predicted class labels by selecting the class with the highest probability
            predicted_labels = np.argmax(probabilities, axis=1)
            # Create a boolean array indicating correct predictions
            correctly_predicted = predicted_labels == labels
            # Update the count of correct predictions
            correct += np.sum(correctly_predicted)

        # Calculate and return the accuracy as the proportion of correct predictions
        return correct / dataset.size


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)
        accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
        writer.add_scalar("dev/accuracy", 100 * accuracy, epoch + 1)

    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy), flush=True)
    writer.add_scalar("test/accuracy", 100 * test_accuracy, epoch + 1)

    return accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
