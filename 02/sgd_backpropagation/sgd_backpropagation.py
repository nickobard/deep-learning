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
            trainable=True
        )
        self._b1 = keras.Variable(keras.ops.zeros([args.hidden_layer]), trainable=True)

        # Creating variables to optimize.
        self._W2 = keras.Variable(keras.random.normal([args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed),
                                  trainable=True)
        self._b2 = keras.Variable(keras.ops.zeros([MNIST.LABELS]), trainable=True)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        # Computation of the network.
        casted_inputs = keras.ops.cast(inputs, dtype="float32")
        normalized = casted_inputs / 255
        prepared_inputs = normalized.reshape(inputs.shape[0], -1)
        hidden_layers = prepared_inputs @ self._W1 + self._b1
        activations = keras.ops.tanh(hidden_layers)
        outputs = activations @ self._W2 + self._b2
        predictions = keras.ops.softmax(outputs)
        return predictions

    def train_epoch(self, dataset: MNIST.Datasplit) -> None:
        for batch in dataset.batches(self._args.batch_size):

            inputs, labels = batch['images'], batch['labels']
            probabilities = self.predict(inputs)

            # Convert labels to one-hot encoding
            labels = keras.ops.one_hot(labels, num_classes=MNIST.LABELS)

            # Compute the average cross-entropy loss over the batch
            loss_manual = -keras.ops.sum(labels * keras.ops.log(probabilities), axis=1)
            loss = keras.ops.mean(loss_manual)

            variables = [self._W1, self._b1, self._W2, self._b2]

            self.zero_grad()
            loss.backward()

            gradients = [variable.value.grad for variable in variables]
            with torch.no_grad():
                for variable, gradient in zip(variables, gradients):
                    # Update variables using SGD with the specified learning rate
                    g_hat = gradient
                    variable.assign_sub(g_hat * self._args.learning_rate)

    def evaluate(self, dataset: MNIST.Datasplit) -> float:
        # Compute the accuracy of the model prediction
        correct = 0
        # Iterate over the dataset in batches
        for batch in dataset.batches(self._args.batch_size):
            inputs, labels = batch['images'], batch['labels']
            # Predict probabilities for each class using the model
            probabilities = keras.ops.convert_to_numpy(self.predict(inputs))

            # Determine the predicted class by selecting the one with the highest probability
            predicted_labels = np.argmax(probabilities, axis=1)
            # Create a boolean array indicating whether each prediction is correct
            correctly_predicted = predicted_labels == labels
            # Update the total count of correct predictions
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

    # Evaluate on test dataset.
    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy), flush=True)
    writer.add_scalar("test/accuracy", 100 * test_accuracy, epoch + 1)

    # Return dev and test accuracies for ReCodEx to validate.
    return accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
