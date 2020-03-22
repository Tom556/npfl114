#!/usr/bin/env bash

source /home/limisiewicz/Documents/Courses/npfl114/dl_venv/bin/activate
python3 ./uppercase.py --hidden_layers "500, 500, 200" \
--embedding_dim 30 --window 3  --alphabet_size 100 \
--decay "exponential" --optimizer "Adam" --learning_rate 0.01 --learning_rate_final 0.0001 \
--dropout 0.4 --l2 1e-5  --clip_norm 0.5 --batch_norm \
--epochs 20 --threads 4