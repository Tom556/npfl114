#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default="none", type=str, help="Activation function.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
    parser.add_argument("--layers", default=1, type=int, help="Number of layers.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the model
    if args.activation == 'none':
        activation = None
    else:
        activation = args.activation

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer([MNIST.H, MNIST.W, MNIST.C]))
    # TODO: Finish the model. Namely add:
    # - a `tf.keras.layers.Flatten()` layer
    # - add `args.layers` number of fully connected hidden layers
    #   `tf.keras.layers.Dense()` with  `args.hidden_layer` neurons, using activation
    #   from `args.activation`, allowing "none", "relu", "tanh", "sigmoid".
    # - finally, add a final fully connected layer with
    #   `MNIST.LABELS` units and `tf.nn.softmax` activation.
    model.add(tf.keras.layers.Flatten(name='flatten'))
    for i in range(args.layers):
        model.add(tf.keras.layers.Dense(args.hidden_layer, activation=activation, name=f"hidden_{i}"))
    model.add(tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax, name="output"))

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    test_logs = model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,
    )
    tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(model.metrics_names, test_logs)})

    accuracy = test_logs[1]
    # TODO: Write test accuracy as percentages rounded to two decimal places.
    with open("mnist_layers_activations.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
