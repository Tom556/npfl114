#!/usr/bin/env bash

python3 ./mnist_layers_activations.py --layers 0 --activation "none"
python3 ./mnist_layers_activations.py --layers 1 --activation "none"
python3 ./mnist_layers_activations.py --layers 1 --activation "relu"
python3 ./mnist_layers_activations.py --layers 1 --activation "tanh"
python3 ./mnist_layers_activations.py --layers 1 --activation "sigmoid"
python3 ./mnist_layers_activations.py --layers 10 --activation "relu"
python3 ./mnist_layers_activations.py --layers 10 --activation "sigmoid"