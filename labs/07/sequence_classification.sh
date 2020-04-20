#!/usr/bin/env bash

python sequence_classification.py --rnn_cell=SimpleRNN --sequence_dim=1
python sequence_classification.py --rnn_cell=SimpleRNN --sequence_dim=2
python sequence_classification.py --rnn_cell=SimpleRNN --sequence_dim=10

python sequence_classification.py --rnn_cell=LSTM --sequence_dim=1
python sequence_classification.py --rnn_cell=LSTM --sequence_dim=2
python sequence_classification.py --rnn_cell=LSTM --sequence_dim=10


python sequence_classification.py --rnn_cell=GRU --sequence_dim=1
python sequence_classification.py --rnn_cell=GRU --sequence_dim=2
python sequence_classification.py --rnn_cell=GRU --sequence_dim=10

python sequence_classification.py --rnn_cell=LSTM --hidden_layer=70 --rnn_cell_dim=30 --sequence_dim=30
python sequence_classification.py --rnn_cell=LSTM --hidden_layer=70 --rnn_cell_dim=30 --sequence_dim=30 --clip_gradient=1

python sequence_classification.py --rnn_cell=SimpleRNN --hidden_layer=70 --rnn_cell_dim=30 --sequence_dim=30
python sequence_classification.py --rnn_cell=SimpleRNN --hidden_layer=70 --rnn_cell_dim=30 --sequence_dim=30 --clip_gradient=1

python sequence_classification.py --rnn_cell=GRU --hidden_layer=90 --rnn_cell_dim=30 --sequence_dim=30
python sequence_classification.py --rnn_cell=GRU --hidden_layer=90 --rnn_cell_dim=30 --sequence_dim=30 --clip_gradient=1