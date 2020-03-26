#!/usr/bin/env bash
python3 ./mnist_regularization.py

python3 ./mnist_regularization.py --dropout 0.3
python3 ./mnist_regularization.py --dropout 0.5
python3 ./mnist_regularization.py --dropout 0.6
python3 ./mnist_regularization.py --dropout 0.8

python3 ./mnist_regularization.py --l2 0.001
python3 ./mnist_regularization.py --l2 0.0001
python3 ./mnist_regularization.py --l2 0.00001

python3 ./mnist_regularization.py --label_smoothing 0.1
python3 ./mnist_regularization.py --label_smoothing 0.3
python3 ./mnist_regularization.py --label_smoothing 0.5

python3 ./mnist_regularization.py --dropout 0.5 --label_smoothing 0.1