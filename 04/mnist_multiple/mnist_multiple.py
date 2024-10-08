#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import numpy as np
import keras
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        images = (
            keras.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            keras.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
        )

        # for both images shared network
        common_subnetwork = [
            keras.layers.Rescaling(1 / 255),
            keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                                activation='relu'),
            keras.layers.Conv2D(filters=20, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                                activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(units=200, activation='relu')
        ]

        # forward pass for each image
        hiddens = []
        for hidden in images:
            for layer in common_subnetwork:
                hidden = layer(hidden)
            hiddens.append(hidden)

        # direct comparison - concatenate features from both images and pass forward to
        # direct comparison subnetwork
        direct_hidden = keras.layers.Concatenate()(hiddens)
        direct_hidden = keras.layers.Dense(units=200, activation='relu')(direct_hidden)
        direct_output = keras.layers.Dense(units=1, activation='sigmoid')(direct_hidden)

        # indirect comparison - get prediction of both images, then compare predicted labels.
        probabilities_layer = keras.layers.Dense(units=MNIST.LABELS, activation='softmax')
        indirect_outputs = [probabilities_layer(hidden) for hidden in hiddens]

        with torch.no_grad():
            digit_pred_1 = keras.ops.argmax(indirect_outputs[0], axis=-1)
            digit_pred_2 = keras.ops.argmax(indirect_outputs[1], axis=-1)
            indirect_comparison = keras.ops.cast(digit_pred_1 > digit_pred_2, "float32")

        # output definition of the model
        outputs = {
            "direct_comparison": direct_output,
            "digit_1": indirect_outputs[0],
            "digit_2": indirect_outputs[1],
            "indirect_comparison": indirect_comparison,
        }

        # Finally, construct the model.
        super().__init__(inputs=images, outputs=outputs)

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        self.compile(
            optimizer=keras.optimizers.Adam(),
            loss={
                "direct_comparison": keras.losses.BinaryCrossentropy(),
                "digit_1": keras.losses.SparseCategoricalCrossentropy(),
                "digit_2": keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "direct_comparison": [keras.metrics.BinaryAccuracy(name='accuracy')],
                "indirect_comparison": [keras.metrics.BinaryAccuracy(name='accuracy')],
            },
        )

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(
            self, mnist_dataset: MNIST.Dataset, args: argparse.Namespace,
    ) -> torch.utils.data.Dataset:
        # Original MNIST dataset.
        images, labels = mnist_dataset.data["images"], mnist_dataset.data["labels"]

        # The new dataset should be created from consecutive _pairs_ of examples.
        # You can assume that the size of the original dataset is even.
        class TorchDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                # half the dataset size, because we take pairs.
                return len(images) // 2

            def __getitem__(self, index: int) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, np.ndarray]]:
                # adjust dataset for our model output, so we can compare true output with predictions.
                image_1, image_2 = images[2 * index], images[2 * index + 1]
                label_1, label_2 = labels[2 * index], labels[2 * index + 1]
                comparison = label_1 > label_2

                output = {
                    "digit_1": label_1,
                    "digit_2": label_2,
                    "direct_comparison": comparison,
                    "indirect_comparison": comparison}

                return (image_1, image_2), output

        return TorchDataset()


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create the model
    model = Model(args)

    # Construct suitable dataloaders from the MNIST data.
    train = torch.utils.data.DataLoader(model.create_dataset(mnist.train, args), args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(model.create_dataset(mnist.dev, args), args.batch_size)

    # Train
    logs = model.fit(train, epochs=args.epochs, validation_data=dev)

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
