#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[400], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay strength.")


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


def main(args: argparse.Namespace) -> dict[str, float]:
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

    # Load data
    mnist = MNIST(size={"train": 5_000})

    # Build the model and incorporate dropout layers
    model = keras.Sequential()
    model.add(keras.layers.Rescaling(1 / 255))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=args.dropout))

    # Add hidden layers with dropout after each Dense layer
    for hidden_layer in args.hidden_layers:
        model.add(keras.layers.Dense(hidden_layer, activation="relu"))
        model.add(keras.layers.Dropout(rate=args.dropout))

    # Add the output layer without dropout
    model.add(keras.layers.Dense(MNIST.LABELS, activation="softmax"))

    # Create an AdamW optimizer with weight decay, excluding biases from decay
    optimizer = keras.optimizers.AdamW(weight_decay=args.weight_decay)
    optimizer.exclude_from_weight_decay(var_names=['bias'])

    # Compile the model with CategoricalCrossentropy loss (supports label smoothing)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    # Initialize TensorBoard callback for logging training progress
    tb_callback = TorchTensorBoardCallback(args.logdir)

    # Train the model using the training data and validate on the development set
    logs = model.fit(
        mnist.train.data["images"], keras.ops.one_hot(mnist.train.data["labels"], num_classes=MNIST.LABELS),
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(
            mnist.dev.data["images"], keras.ops.one_hot(mnist.dev.data["labels"], num_classes=MNIST.LABELS)),
        callbacks=[tb_callback],
    )

    # Return development metrics for validation
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
