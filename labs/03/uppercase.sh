#!/usr/bin/env bash

source /home/limisiewicz/Documents/Courses/npfl114/dl_venv/bin/activate
python3 ./uppercase.py  --activation "relu" --recurrence "LSTM"  --hidden_layers "100" --recurrence_layers "200" \
--window 3  --alphabet_size 60 \
--decay "exponential" --optimizer "Adam" --learning_rate 0.05 --learning_rate_final 0.001 \
--dropout 0.4 --label_smoothing=0.001 --class_weights \
--epochs 30 --threads 4