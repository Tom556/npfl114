#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    # TODO: Set reasonable defaults and possibly add more arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default="none", type=str, help="Activation function.")
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--model", default="gym_cartpole_model.h5", type=str, help="Output model path.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
    parser.add_argument("--layers", default=1, type=int, help="Number of layers.")
    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
    parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")
    parser.add_argument("--l2_penalty", default=0.0, type=float, help="L2 penalty for regularization in hidden layers")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    observations, labels = [], []
    with open("gym_cartpole-data.txt", "r") as data:
        for line in data:
            columns = line.rstrip("\n").split()
            observations.append([float(column) for column in columns[:-1]])
            labels.append(int(columns[-1]))
    observations, labels = np.array(observations), np.array(labels)

    # TODO: Create the model in the `model` variable. Note that
    # the model can perform any of:
    # - binary classification with 1 output and sigmoid activation;
    # - two-class classification with 2 outputs and softmax activation.

    if args.activation == 'none':
        activation = None
    else:
        activation = args.activation

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer([4]))

    model.add(tf.keras.layers.Flatten(name='flatten'))
    for i in range(args.layers):
        model.add(tf.keras.layers.Dense(args.hidden_layer, activation=activation,
                                        kernel_regularizer=tf.keras.regularizers.l2(args.l2_penalty),name=f"hidden_{i}"))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="output"))

    # TODO: Prepare the model for training using the `model.compile` method.
    learning_rate_final = args.learning_rate_final

    decay_steps = int(observations.shape[0] * args.epochs / args.batch_size)
    learning_rate_schedule = None
    if args.decay == 'polynomial':
        learning_rate_schedule = tf.optimizers.schedules.PolynomialDecay(args.learning_rate,
                                                decay_steps=decay_steps, end_learning_rate=learning_rate_final)
    elif args.decay == 'exponential':
        decay_rate = learning_rate_final / args.learning_rate
        learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                 decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
    else:
       learning_rate_schedule = args.learning_rate

    optimizer = None
    if args.optimizer == 'SGD':
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate_schedule, momentum=(args.momentum or 0.0))
    elif args.optimizer == 'Adam':
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule)

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)

    model.fit(
        observations, labels,
        batch_size=args.batch_size, epochs=args.epochs,
        callbacks=[tb_callback], validation_split=0.2
    )

    # Save the model, without the optimizer state.
    model.save(args.model, include_optimizer=False)
