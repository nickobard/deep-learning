#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, choices=["linear", "exponential", "cosine"], help="Decay type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=128, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"], help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


# If you add more arguments, ReCodEx will keep them with your default values.


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
    mnist = MNIST()

    # Create the model
    model = keras.Sequential([
        keras.layers.Rescaling(1 / 255),
        keras.layers.Flatten(),
        keras.layers.Dense(args.hidden_layer, activation="relu"),
        keras.layers.Dense(MNIST.LABELS, activation="softmax"),
    ])

    model.summary()

    if args.decay == 'linear':
        max_steps = (mnist.train.size // args.batch_size) * args.epochs
        learning_rate = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=args.learning_rate,
            end_learning_rate=args.learning_rate_final,
            decay_steps=max_steps
        )
    elif args.decay == 'exponential':
        max_steps = (mnist.train.size // args.batch_size) * args.epochs
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_rate=args.learning_rate_final / args.learning_rate,
            decay_steps=max_steps
        )
    elif args.decay == 'cosine':
        max_steps = (mnist.train.size // args.batch_size) * args.epochs
        cosine_decay = 0.5 * (1 + np.cos(np.pi))
        alpha = (args.learning_rate_final / args.learning_rate) / (1 - cosine_decay) - cosine_decay / (1 - cosine_decay)
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            alpha=alpha,
            decay_steps=max_steps
        )
    else:
        learning_rate = args.learning_rate

    if args.optimizer == 'SGD':
        if args.momentum is not None:
            kwargs = {'momentum': args.momentum, 'nesterov': True}
        else:
            kwargs = {}

        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, **kwargs)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    tb_callback = TorchTensorBoardCallback(args.logdir)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
