#!/usr/bin/env bash

source /home/limisiewicz/Documents/Courses/npfl114/dl_venv/bin/activate
python3 ./uppercase.py --recurrence "LSTM" --hidden_layers "100" --recurrence_layers "200" \
--window 3  --alphabet_size 60 \
--decay "exponential" --optimizer "Adam" --learning_rate 0.01 --learning_rate_final 0.0001 \
--dropout 0.4 --l2=0.02  --clip_norm=1. \
--epochs 20 --threads 4