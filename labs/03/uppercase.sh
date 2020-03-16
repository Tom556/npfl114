#!/usr/bin/env bash

python3 ./uppercase.py  --activation "relu" --hidden_layers "100" --window 3 --epochs 1 --alphabet_size 100 --class_weights "1., 3." --deacy "exponential" --optimizer "Adam" --learning_rate 0.05 --learning_rate_final 0.0001
