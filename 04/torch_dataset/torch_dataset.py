#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import numpy as np
import keras
import torch
from torchvision.transforms import v2

from cifar10 import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("--augment", default=False, action="store_true", help="Whether to augment the data.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--show_images", default=None, const=10, type=int, nargs="?", help="Show augmented images.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load the data
    cifar = CIFAR10()

    # Create the model
    inputs = keras.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
    hidden = keras.layers.Rescaling(1 / 255)(inputs)
    hidden = keras.layers.Conv2D(16, 3, 2, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(16, 3, 1, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(24, 3, 2, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(24, 3, 1, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(32, 3, 2, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(32, 3, 1, "same", activation="relu")(hidden)
    hidden = keras.layers.Flatten()(hidden)
    hidden = keras.layers.Dense(200, activation="relu")(hidden)
    outputs = keras.layers.Dense(len(CIFAR10.LABELS), activation="softmax")(hidden)

    # Compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, cifar: CIFAR10.Dataset, size: int, augmentation_fn=None) -> None:
            self._size = min(cifar.size, size)
            self._images = cifar.data['images'][:self._size]
            self._labels = cifar.data['labels'][:self._size]
            self._augmentation_fn = augmentation_fn

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index: int) -> tuple[np.ndarray | torch.Tensor, int]:
            if self._augmentation_fn:
                return self._augmentation_fn(self._images[index]), self._labels[index]
            return self._images[index], self._labels[index]

    if args.augment:
        transformation = v2.Compose([
            v2.RandomResize(min_size=28, max_size=36),
            v2.Pad(padding=4),
            v2.RandomCrop(size=(32, 32)),
            v2.RandomHorizontalFlip()
        ])

        def augmentation_fn(image: np.ndarray) -> torch.Tensor:
            pt_image = torch.from_numpy(image).type(torch.uint8)
            pt_image_chw = pt_image.permute(2, 0, 1)
            transformed_chw = transformation(pt_image_chw)
            transformed_hwc = transformed_chw.permute(1, 2, 0)
            return transformed_hwc
    else:
        augmentation_fn = None

    train = TorchDataset(cifar.train, 5000, augmentation_fn=augmentation_fn)
    dev = TorchDataset(cifar.dev, 1000)

    if args.show_images:
        from torch.utils import tensorboard
        GRID, REPEATS, TAG = args.show_images, 5, "augmented" if args.augment else "original"
        tb_writer = tensorboard.SummaryWriter(os.path.join("logs", "augmentations"))
        for step in range(REPEATS):
            images = keras.ops.stack([train[i][0] for i in range(GRID * GRID)], axis=0)
            images = images.reshape(GRID, GRID * images.shape[1], *images.shape[2:]).permute(0, 2, 1, 3)
            images = images.reshape(1, GRID * images.shape[1], *images.shape[2:]).permute(0, 2, 1, 3)
            tb_writer.add_images(TAG, images, step, dataformats="NHWC")
        tb_writer.close()
        print("Saved first {} training imaged to logs/{}".format(GRID * GRID, TAG))

    train = torch.utils.data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dataset=dev, batch_size=args.batch_size, shuffle=True)

    # Train
    logs = model.fit(train, epochs=args.epochs, validation_data=dev)

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
