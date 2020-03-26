#!/usr/bin/env bash

python3 ./mnist_training.py --optimizer "Adam" --learning_rate 0.01 --decay "exponential" --learning_rate_final 0.001 --layers 10 --activation "relu"
# best with relu
python3 ./gym_cartpole.py --optimizer "Adam" --learning_rate 0.01 --decay "exponential" --learning_rate_final 0.0001 --layers 5 --hidden_layer 100  --batch_size 5  --activation "relu" --epochs 400

# best 499
python3 ./gym_cartpole.py --optimizer "Adam" --learning_rate 0.01 --decay "exponential" --learning_rate_final 0.0001 --layers 3 --hidden_layer 100  --batch_size 5  --activation "tanh" --epochs 100 --l2_penalty 0.001
# decreased number of epochs to 50 and the score is 500