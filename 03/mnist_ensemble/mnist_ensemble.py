#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--models", default=3, type=int, help="Number of models.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create models
    models = []
    for model in range(args.models):
        models.append(keras.Sequential([
                                           keras.layers.Rescaling(1 / 255),
                                           keras.layers.Flatten(),
                                       ] + [keras.layers.Dense(hidden_layer, activation="relu") for hidden_layer in
                                            args.hidden_layers] + [
                                           keras.layers.Dense(MNIST.LABELS, activation="softmax"),
                                       ]))

        models[-1].compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        print("Training model {}: ".format(model + 1), end="", flush=True)
        models[-1].fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs, verbose=0
        )
        print("Done")

    individual_accuracies, ensemble_accuracies = [], []
    for model in range(args.models):
        # Retrieve the dev set images and labels
        X = mnist.dev.data["images"]
        y = mnist.dev.data["labels"]

        # Initialize the accuracy metric
        cat_acc = keras.metrics.SparseCategoricalAccuracy()

        # Compute accuracy of the individual model on the dev set
        cat_acc.update_state(y_true=y, y_pred=models[model].predict(X))
        individual_accuracy = cat_acc.result()
        cat_acc.reset_state()

        # Compute accuracy of the ensemble of models up to the current one
        # Collect predictions from all models in the ensemble
        predictions = []
        for m in models[0:model + 1]:
            predictions.append(m.predict(X))

        # Average the predictions across the ensemble
        mean_predictions = keras.ops.mean(predictions, axis=0)
        # Update the accuracy metric with the averaged predictions
        cat_acc.update_state(y_true=y, y_pred=mean_predictions)

        # Store the accuracies
        individual_accuracies.append(individual_accuracy)
        ensemble_accuracies.append(cat_acc.result())
    return individual_accuracies, ensemble_accuracies


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    individual_accuracies, ensemble_accuracies = main(args)
    for model, (individual_accuracy, ensemble_accuracy) in enumerate(zip(individual_accuracies, ensemble_accuracies)):
        print("Model {}, individual accuracy {:.2f}, ensemble accuracy {:.2f}".format(
            model + 1, 100 * individual_accuracy, 100 * ensemble_accuracy))
