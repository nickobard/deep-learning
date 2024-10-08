#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch

from uppercase_data import UppercaseData

parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=100, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=4, type=int, help="Window size to use.")

parser.add_argument("--activation", default="relu", choices=["none", "relu", "tanh", "sigmoid"], help="Activation.")
parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")


class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard
            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(os.path.join(self._path, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                logs = logs | {"learning_rate": keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)}
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            self.add_logs("val", {k[4:]: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the uppercase data with specified window size and alphabet size
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    # Build the model for character capitalization prediction
    activation = args.activation if args.activation != "none" else None

    model = keras.models.Sequential([
                                        # Input layer expecting sequences of character indices
                                        keras.layers.Input(shape=[2 * args.window + 1], dtype="int32"),
                                        # Convert character indices to one-hot encoding (very important, otherwise won't work)
                                        keras.layers.CategoryEncoding(args.alphabet_size, output_mode="one_hot"),
                                        # Flatten the one-hot encoded inputs to a single vector
                                        keras.layers.Flatten()] + [
                                        # Add hidden Dense layers with specified activation function
                                        keras.layers.Dense(units=hidden_layer, activation=activation) for
                                        hidden_layer in args.hidden_layers] + [
                                        # Output layer with a sigmoid activation for binary classification
                                        keras.layers.Dense(1, activation='sigmoid')
                                    ])
    # Compile the model with Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="Accuracy")]
    )

    model.summary()

    tb_callback = TorchTensorBoardCallback(args.logdir)

    # Train the model on the training data and validate on the development set
    model.fit(x=uppercase_data.train.data['windows'], y=uppercase_data.train.data['labels'], batch_size=args.batch_size,
              epochs=args.epochs, callbacks=[tb_callback],
              validation_data=(uppercase_data.dev.data['windows'], uppercase_data.dev.data['labels']))

    # Generate the correctly capitalized test set
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict capitalization probabilities for the test data windows
        prediction = model.predict(uppercase_data.test.data['windows'])
        # Iterate over the original test text characters
        for index, character in enumerate(uppercase_data.test.text):
            # If the predicted probability is greater than 0.5, capitalize the character
            predictions_file.write(str.upper(character) if prediction[index] > 0.5 else character)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
