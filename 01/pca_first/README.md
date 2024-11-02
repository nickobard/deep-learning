
# PCA First Assignment

This project explores PyTorch tensor manipulation and principal component analysis (PCA) on the MNIST dataset, focusing on computing the covariance matrix, identifying the first principal component, and quantifying its explained variance. This assignment builds a foundation in PyTorch tensors, shapes, and tensor operations.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

In this assignment:
1. **Load the MNIST Dataset**: Using the `MNIST` class in `mnist.py`, load a subset of MNIST examples.
2. **Compute Covariance and PCA**:
    - Reshape the images to a 1D format.
    - Compute the covariance matrix.
    - Run a power iteration algorithm to find the first principal component and the explained variance.
3. **Output**:
    - **Total Variance**: The sum of variances (diagonal elements) in the covariance matrix.
    - **Explained Variance**: The variance explained by the first principal component as a percentage of the total variance.

**Example Commands**:
```bash
python3 pca_first.py --examples=1024 --iterations=64
```

**Expected Output Format**:
```plaintext
Total variance: X.XX
Explained variance: X.XX%
```

## Solution Explanation

The solution (`pca_first.py`) includes:
1. **Argument Parsing**: Command-line arguments specify the number of examples and iterations for the power algorithm.
2. **Data Loading**: `MNIST` class in `mnist.py` loads and normalizes data.
3. **Tensor Reshaping**: Images are reshaped from 28x28 format to a 1D array (784 elements).
4. **Covariance Calculation**:
    - Compute the covariance matrix using the mean-centered data.
    - Calculate total variance as the sum of diagonal elements of the covariance matrix.
5. **Power Iteration for PCA**:
    - Find the dominant eigenvector using the power method, iterating as specified in `--iterations`.
    - Calculate explained variance by dividing the dominant eigenvalue by the total variance.

## Usage

Run the script with specific arguments, like:
```bash
python3 pca_first.py --examples=<number_of_examples> --iterations=<number_of_iterations>
```

Example:
```bash
python3 pca_first.py --examples=1024 --iterations=64
```

## File Structure

The files in this project are as follows:
- `pca_first.py`: Main script for PCA computation on MNIST data.
- `mnist.py`: Utility script to load and process MNIST data.
- `README.md`: Project documentation.
