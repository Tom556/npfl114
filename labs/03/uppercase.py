#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

if __name__ == "__main__":
    # Parse arguments
    # TODO: Set reasonable values for `alphabet_size` and `window`.
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet_size", default=None, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--embedding_dim", default=15, type=int, help="Dimensionality of embeddings.")
    parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--window", default=None, type=int, help="Window size to use.")
    parser.add_argument("--activation", default="none", type=str, help="Activation function.")
    parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
    parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
    parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")
    parser.add_argument("--class_weights", default="0.5, 0.5", type=str, help="Weights for the classes.")
    parser.add_argument("--recurrence", default=None, type=str, help="Type of recurrent network [LSTM, GRU]")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]
    args.class_weights = {idx: float(weight) for idx, weight in enumerate(args.class_weights.split(",")) if weight}
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

    # Load data
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on left,
    # the character in question, and `args.window` characters on right), where
    # each character is representedy by a `tf.int32` index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_. There is no
    #   explicit Keras layer, but you can
    #   - use a Lambda layer which can encompass any function:
    #       tf.keras.Sequential([
    #         tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.

    if args.activation == 'none':
        activation = None
    else:
        activation = args.activation

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(args.alphabet_size, args.embedding_dim, input_length=2 * args.window + 1))

    if args.recurrence not in ["LSTM", "GRU"]:
        model.add(tf.keras.layers.Flatten(name='flatten'))

    for i, layer_dim in enumerate(args.hidden_layers):
        if args.recurrence == "LSTM":
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                layer_dim, kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                return_sequences=(i != len(args.hidden_layers) - 1), name=f"biLSTM_{i}")))

        elif args.recurrence == "GRU":
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                layer_dim, kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                return_sequences=(i != len(args.hidden_layers) - 1), name=f"biGRU_{i}")))

        else:
            model.add(tf.keras.layers.Dense(layer_dim, activation=activation,
                                            kernel_regularizer=tf.keras.regularizers.l2(args.l2), name=f"hidden_{i}"))

    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="output"))

    # TODO: Prepare the model for training using the `model.compile` method.
    learning_rate_final = args.learning_rate_final

    decay_steps = int(uppercase_data.train.size * args.epochs / args.batch_size)
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
        uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
        #class_weight=args.class_weights,
        callbacks=[tb_callback]
    )

    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as out_file:
        # TODO: Generate correctly capitalized test set.
        # Use `uppercase_data.test.text` as input, capitalize suitable characters,
        # and write the result to `uppercase_test.txt` file.
        predictions = model.predict(uppercase_data.test.data["windows"])
        for pred, char in zip(predictions, uppercase_data.test.text):
            #char = uppercase_data.test.alphabet[window[args.window]]
            if pred[1] > pred[0]:
                out_file.write(char.upper())
            else:
                out_file.write(char)
    out_file.close()

