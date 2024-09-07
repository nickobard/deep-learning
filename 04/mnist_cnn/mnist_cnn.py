#!/usr/bin/env python3
import argparse
import os
from argparse import ArgumentError

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch
import re

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def add_convolution_layer(last_layer, hparams, use_batch_norm: bool = False):
    if use_batch_norm:
        conv_layer = keras.layers.Conv2D(**hparams,
                                         activation=None, use_bias=False)(last_layer)
        batch_norm_layer = keras.layers.BatchNormalization()(conv_layer)
        relu_layer = keras.layers.ReLU()(batch_norm_layer)
        return relu_layer
    else:
        return keras.layers.Conv2D(**hparams,
                                   activation="relu")(last_layer)


def parse_and_add_convolution_layer(hparams_specs: str, last_layer):
    # parse and extract hyperparameters
    filters = int(hparams_specs[1])
    kernel_size = int(hparams_specs[2])
    stride = int(hparams_specs[3])
    padding = hparams_specs[4]
    hparams = {'filters': filters, 'kernel_size': (kernel_size, kernel_size),
               'strides': (stride, stride), 'padding': padding}
    # split depending if to use batch normalization or not
    if hparams_specs[0].endswith('B'):
        return add_convolution_layer(last_layer, hparams, use_batch_norm=True)
    else:
        return add_convolution_layer(last_layer, hparams)


class Model(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Functional API
        # Input layer, used later to init model
        inputs = keras.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        # hidden layer variable to store the last layer with all layers history
        hidden = keras.layers.Rescaling(1 / 255)(inputs)

        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer **without bias** and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.

        # Regular expression to match every specification. Split by comma cannot be used because of the
        # residual layers specification, may cantain commas internally.
        layer_args = re.findall(r'[^,]*\[.*\]|[^,]+', args.cnn)

        for layer in layer_args:
            hparams = layer.split('-')
            if hparams[0].startswith('C'):
                hidden = parse_and_add_convolution_layer(hparams, hidden)
            elif hparams[0] == 'M':
                pool_size = int(hparams[1])
                stride = int(hparams[2])
                hidden = keras.layers.MaxPool2D(pool_size=(pool_size, pool_size), strides=(stride, stride),
                                                padding="valid")(hidden)
            elif hparams[0] == "R":
                # save reference to layer that is input to residual layers
                residual_input = hidden
                # separate residual layers specification from the whole specification.
                rl_layers = re.findall(pattern=r'\[([^\[\]]*)\]', string=layer)[0].split(',')
                # for each layer in residual layers
                for residual_layer in rl_layers:
                    rl_hparams = residual_layer.split("-")
                    if rl_hparams[0].startswith('C'):
                        hidden = parse_and_add_convolution_layer(rl_hparams, hidden)
                    else:
                        ArgumentError(argument=residual_layer, message=f"Unexpected cnn argument: {layer}")
                # add input to residual layers
                hidden = hidden + residual_input
            elif hparams[0] == "F":
                hidden = keras.layers.Flatten()(hidden)
            elif hparams[0] == "H":
                hidden_layer_size = hparams[1]
                hidden = keras.layers.Dense(units=hidden_layer_size, activation="relu")(hidden)
            elif hparams[0] == "D":
                dropout_rate = float(hparams[1])
                hidden = keras.layers.Dropout(dropout_rate)(hidden)
            else:
                raise ArgumentError(argument=layer, message=f"Unexpected cnn argument: {layer}")

        # Add the final output layer
        outputs = keras.layers.Dense(MNIST.LABELS, activation="softmax")(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.summary()


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
