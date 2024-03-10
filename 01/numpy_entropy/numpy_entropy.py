#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data_2.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model_2.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")


# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    data_frequencies = {}
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            if line in data_frequencies:
                data_frequencies[line] += 1
            else:
                data_frequencies[line] = 1

    outcomes_number = sum(data_frequencies.values())
    data_probabilities = {outcome: frequency / outcomes_number for outcome, frequency in data_frequencies.items()}
    data_outcomes = data_probabilities.keys()

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.

    data_distribution = np.array([data_probabilities[outcome] for outcome in sorted(data_probabilities)])

    # TODO: Load model distribution, each line `string \t probability`.

    model_probabilities = {}

    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating using Python data structures.

            outcome, probability = line.split()
            model_probabilities[outcome] = float(probability)

    model_outcomes = model_probabilities.keys()

    # TODO: Create a NumPy array containing the model distribution.
    model_distribution = np.array(
        [model_probabilities[outcome] for outcome in sorted(model_outcomes) if
         outcome in data_outcomes])

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -(sum(data_distribution * np.log(data_distribution)))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.

    missing_in_model = data_outcomes - model_outcomes

    if len(missing_in_model) == 0:
        crossentropy = -sum(data_distribution * np.log(model_distribution))
    else:
        crossentropy = np.inf

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.

    if len(missing_in_model) == 0:
        kl_divergence = sum(data_distribution * (np.log(data_distribution) - np.log(model_distribution)))
    else:
        kl_divergence = np.inf

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))