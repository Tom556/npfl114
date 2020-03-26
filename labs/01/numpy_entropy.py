#!/usr/bin/env python3
import numpy as np
from collections import defaultdict

if __name__ == "__main__":
    # Load data distribution, each line containing a datapoint -- a string.
    label_count = defaultdict(int)
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            label_count[line] += 1


    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.

    # Load model distribution, each line `string \t probability`.

    label_prob = defaultdict(float)
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            line = line.split("\t")
            label_prob[line[0]] = float(line[1])

    # TODO: Create a NumPy array containing the model distribution.
    all_labels = list(label_count.keys())

    data_probs = np.array([label_count[label] for label in all_labels])
    data_probs = data_probs / np.sum(data_probs)
    model_probs = np.array([label_prob[label] for label in all_labels])
    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(data_probs * np.log(data_probs))
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)

    cross_entropy = -np.sum(data_probs * np.log(model_probs))
    print("{:.2f}".format(cross_entropy))
    D_KL = cross_entropy - entropy
    print("{:.2f}".format(D_KL))
