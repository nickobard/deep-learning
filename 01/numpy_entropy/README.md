# Numpy Entropy Assignment

This project aims to familiarize you with Python, NumPy, and the ReCodEx submission system by implementing entropy, cross-entropy, and KL-divergence calculations based on probability distributions. The solution leverages `numpy` for vectorized operations, calculating the three measures and printing the results.

## Table of Contents
- [Assignment Overview](#assignment-overview)
- [Solution Explanation](#solution-explanation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Assignment Overview

The task involves:
1. **Loading Data Files**: Reading files specified in `args.data_path` and `args.model_path`, containing data points and model probability distributions, respectively.
2. **Calculating Metrics**:
    - **Entropy** of the data distribution.
    - **Cross-entropy** between data and model distributions.
    - **KL-Divergence** from data distribution to model distribution.
3. **Printing Results**: Each value is printed on a new line, rounded to two decimal places, with "inf" for infinite values (when the model distribution assigns zero probability to data points).

**Example Commands**:
```bash
python3 numpy_entropy.py --data_path numpy_entropy_data_1.txt --model_path numpy_entropy_model_1.txt
```

**Expected Output Format**:
```plaintext
Entropy: X.XX nats
Crossentropy: X.XX nats
KL divergence: X.XX nats
```

## Solution Explanation

The solution (`numpy_entropy.py`) includes:
1. **Parsing Arguments**: Command-line arguments `data_path` and `model_path` are used to specify data files.
2. **Loading and Processing Data**:
    - **Data Distribution**: Counts occurrences of each data point in `data_path`, calculates probabilities, and stores them in a NumPy array.
    - **Model Distribution**: Loads the model probability distribution from `model_path` as a NumPy array.
3. **Calculating Metrics**:
    - **Entropy** is computed using `-sum(data_distribution * np.log(data_distribution))`.
    - **Cross-entropy** is calculated only if all data points exist in the model distribution, otherwise set to `np.inf`.
    - **KL-Divergence** is computed similarly, returning `np.inf` if any data points are missing in the model distribution.
4. **Output**: Results are printed in the specified format, rounded to two decimal places.

## Usage

Run the script using:
```bash
python3 numpy_entropy.py --data_path <data_file> --model_path <model_file>
```

Example:
```bash
python3 numpy_entropy.py --data_path numpy_entropy_data_1.txt --model_path numpy_entropy_model_1.txt
```

## File Structure

The files included in this project:
- `numpy_entropy.py`: Main script to calculate entropy, cross-entropy, and KL-divergence.
- `numpy_entropy_data_*.txt`: Data files for testing.
- `numpy_entropy_model_*.txt`: Model files for testing.
- `README.md`: Assignment documentation.
