#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

class Model(tf.Module):
    def __init__(self, args):
        self._W1 = tf.Variable(tf.random.normal([MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed), trainable=True)
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)

        # TODO(sgd_backpropagation): Create variables:
        # - _W2, which is a trainable Variable of size [args.hidden_layer, MNIST.LABELS],
        #   initialized to `tf.random.normal` value with stddev=0.1 and seed=args.seed,
        # - _b2, which is a trainable Variable of size [MNIST.LABELS] initialized to zeros

        self._W2 = tf.Variable(tf.random.normal([args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed), trainable=True)
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]), trainable=True)

    def predict(self, inputs):
        # TODO(sgd_backpropagation): Define the computation of the network. Notably:
        # - start by reshaping the inputs to shape [inputs.shape[0], -1].
        #   The -1 is a wildcard which is computed so that the number
        #   of elements before and afte the reshape fits.
        # - then multiply the inputs by `self._W1` and then add `self._b1`
        # - apply `tf.nn.tanh`
        # - multiply the result by `self._W2` and then add `self._b2`
        # - finally apply `tf.nn.softmax` and return the result

        inputs = tf.reshape(inputs, [inputs.shape[0], -1])
        latents = inputs @ self._W1 + self._b1
        latents = tf.nn.tanh(latents)
        outputs = latents @ self._W2 + self._b2
        outputs = tf.nn.softmax(outputs)

        # TODO: In order to support manual gradient computation, you should
        # return not only the output layer, but also the hidden layer after applying
        # tf.nn.tanh, and the input layer after reshaping.

        return inputs, latents, outputs

    def train_epoch(self, dataset):
        for batch in dataset.batches(args.batch_size):
            # The batch contains
            # - batch["images"] with shape [?, MNIST.H, MNIST.W, MNIST.C]
            # - batch["labels"] with shape [?]
            # Size of the batch is `args.batch_size`, except for the last, which
            # might be smaller.

            # TODO: Contrary to sgd_backpropagation, the goal here is to compute
            # the gradient manually, without tf.GradientTape. ReCodEx checks
            # this by searching for "GradientTape" outside any comment block, and
            # if such "GradientTape" is found, the solution does not pass.

            # TODO: Compute the input layer, hidden layer and output layer
            # of the batch images using `self.predict`.

            inputs, latents, outputs = self.predict(batch["images"])

            # TODO: Compute the loss:
            # - for every batch example, it is the categorical crossentropy of the
            #   predicted probabilities and gold batch label
            # - finally, compute the average across the batch examples
            one_hot_labels = tf.one_hot(batch["labels"], MNIST.LABELS)
            loss = tf.losses.categorical_crossentropy(one_hot_labels, outputs)
            loss = tf.reduce_mean(loss)

            # TODO: Compute the gradient of the loss with respect to all
            # variables. In order to compute an outer product
            #   `C[a, i, j] = A[a, i] * B[a, j]`
            # you can use for example either
            #   `A[:, :, tf.newaxis] * B[:, tf.newaxis, :]`
            # or
            #   `tf.einsum("ai,aj->aij", A, B)`

            output_gradients = -one_hot_labels + outputs
            W2_gradients = latents[:, :, tf.newaxis] * output_gradients[:, tf.newaxis, :]
            b2_gradients = output_gradients

            # back propagated gradient tense derivative of the tanh
            latent_gradients = tf.reduce_sum(self._W2 * output_gradients[:, tf.newaxis, :], axis=2) * (1 - latents * latents)

            W1_gradients = inputs[:, :, tf.newaxis] * latent_gradients[:, tf.newaxis, :]
            b1_gradients = latent_gradients

            # TODO(sgd_backpropagation): Perform the SGD update with learning rate `args.learning_rate`
            # for the variable and computed gradient. You can modify
            # variable value with `variable.assign` or in this case the more
            # efficient `variable.assign_sub`.

            self._W2.assign_sub(args.learning_rate * tf.reduce_mean(W2_gradients, axis=0))
            self._b2.assign_sub(args.learning_rate * tf.reduce_mean(b2_gradients, axis=0))
            self._W1.assign_sub(args.learning_rate * tf.reduce_mean(W1_gradients, axis=0))
            self._b1.assign_sub(args.learning_rate * tf.reduce_mean(b1_gradients, axis=0))

    def evaluate(self, dataset):
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(args.batch_size):
            # TODO(sgd_backpropagation): Compute the probabilities of the batch images
            _, _, probabilities = self.predict(batch["images"])
            # TODO: Evaluate how many batch examples were predicted
            # correctly and increase `correct` variable accordingly.
            correct += tf.reduce_sum(tf.cast(tf.equal(tf.argmax(probabilities, axis=1), batch["labels"]), dtype=tf.float32))
        return correct / dataset.size


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
    parser.add_argument("--learning_rate", default=0.2, type=float, help="Learning rate.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
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

    # Load data
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10*1000)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        # TODO: Run the `train_epoch` with `mnist.train` dataset
        model.train_epoch(mnist.train)
        # TODO: Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        accuracy = model.evaluate(mnist.dev)

        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
        with writer.as_default():
            tf.summary.scalar("dev/accuracy", 100 * accuracy, step=epoch + 1)

    # TODO: Evaluate the test data using `evaluate` on `mnist.test` dataset
    accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
    with writer.as_default():
        tf.summary.scalar("test/accuracy", 100 * accuracy, step=epoch + 1)

    # Save the test accuracy in percents rounded to two decimal places.
    with open("sgd_manual.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
