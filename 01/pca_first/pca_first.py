#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
parser.add_argument("--examples", default=256, type=int, help="MNIST examples to use.")
parser.add_argument("--iterations", default=100, type=int, help="Iterations of the power algorithm.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Set the random seed and the number of threads.
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    data_indices = np.random.choice(mnist.train.size, size=args.examples, replace=False)
    data = torch.tensor(mnist.train.data["images"][data_indices] / 255, dtype=torch.float32)

    # Data has shape [args.examples, MNIST.H, MNIST.W, MNIST.C].
    # Reshape it to [args.examples, MNIST.H * MNIST.W * MNIST.C].
    reshaped = torch.reshape(data, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))
    # reshaped shape should be torch.Size([256, 784])

    # Compute the mean of every feature.
    # The `dim=0` argument was used to compute the mean across the first dimension (across examples).
    mean = torch.mean(reshaped, dim=0)

    # Compute the covariance matrix.
    cov = (reshaped - mean).T @ (reshaped - mean) / reshaped.shape[0]

    # Compute the total variance as the sum of variances (diagonal elements of the covariance matrix).
    total_variance = torch.sum(torch.diagonal(cov))

    # Run the power iteration algorithm to find the dominant eigenvector
    v = torch.ones(cov.shape[0], dtype=torch.float32)
    for i in range(args.iterations):
        v = cov @ v
        s = v.norm(p=2)
        v = v / s

    # The `v` is now approximately the eigenvector of the largest eigenvalue, `s`.
    # We now compute the explained variance, which is the ratio of `s` and `total_variance`.
    explained_variance = s / total_variance

    return total_variance, 100 * explained_variance


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    total_variance, explained_variance = main(args)
    print("Total variance: {:.2f}".format(total_variance))
    print("Explained variance: {:.2f}%".format(explained_variance))
