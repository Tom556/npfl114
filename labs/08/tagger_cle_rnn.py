#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, args, num_words, num_tags, num_chars):
        # TODO(we): Implement a one-layer RNN network. The input
        # `word_ids` consists of a batch of sentences, each
        # a sequence of word indices. Padded words have index 0.
        word_ids = tf.keras.layers.Input(shape=[None])

        # TODO(we): Embed input words with dimensionality `args.we_dim`,
        # using `mask_zero=True`.

        wle = tf.keras.layers.Embedding(num_words, args.we_dim, mask_zero=True)(word_ids)

        # TODO: The RNN character-level embeddings utilize the input `charseqs`
        # containing a sequence of character indices for every input word.
        # Again, padded characters have index 0.

        charseqs = tf.keras.layers.Input(shape=[None, None])

        # Because cuDNN implementation of RNN does not allow empty sequences,
        # we need to consider only charseqs for valid words.
        valid_words = tf.where(word_ids != 0)
        cle = tf.gather_nd(charseqs, valid_words)

        # TODO: Embed the characters in `cle` using embeddings of size
        # `args.cle_dim`, masking zero indices. Then, pass the embedded characters
        # through a bidirectional GRU with dimension `args.cle_dim`, concatenating
        # results from forward and backward pass. Store the computed embeddings
        # in `cle` variable.

        cle = tf.keras.layers.Embedding(num_chars, args.cle_dim, mask_zero=True)(cle)

        cle = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(args.cle_dim, return_sequences=False, name="CL_GRU"), merge_mode='concat')(cle)

        # Now we copy cle-s back to the original shape.
        cle = tf.scatter_nd(valid_words, cle, [tf.shape(charseqs)[0], tf.shape(charseqs)[1], cle.shape[-1]])

        # TODO: Concatenate the WE and CLE embeddings (in this order).
        # Use a `tf.keras.layers.Concatenate()` layer, which preserves masks
        # (contrary to raw methods like tf.concat).

        x = tf.keras.layers.Concatenate()([wle, cle])

        # TODO(we): Create specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim` and apply it in a bidirectional way on
        # the embedded words, summing the outputs of forward and backward RNNs.

        if args.rnn_cell == 'LSTM':
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True, name="LSTM"), merge_mode='sum')(x)
        elif args.rnn_cell == 'GRU':
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(args.rnn_cell_dim, return_sequences=True, name="GRU"), merge_mode='sum')(x)
        else:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.SimpleRNN(args.rnn_cell_dim, return_sequences=True, name="GRU"), merge_mode='sum')(x)

        # TODO(we): Add a softmax classification layer into `num_tags` classes, storing
        # the outputs in `predictions`.

        predictions = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_tags, activation='softmax'))(x)

        self.model = tf.keras.Model(inputs=[word_ids, charseqs], outputs=predictions)
        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.losses.SparseCategoricalCrossentropy(),
                           metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            # TODO: Train with the given batch, using `train_on_batch`, with
            # - [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseqs] as inputs
            # - batch[dataset.TAGS].word_ids as targets
            # Additionally, pass `reset_metrics=True`.
            #
            # Store the computed metrics in `metrics`.
            # word_ids = tf.keras.preprocessing.sequence.pad_sequences(batch[dataset.FORMS].word_ids, padding='post')
            # charseqs = tf.keras.preprocessing.sequence.pad_sequences(batch[dataset.FORMS].word_ids, padding='post')

            metrics = self.model.train_on_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseqs],
                                                batch[dataset.TAGS].word_ids,
                                                reset_metrics=True)

            # Generate the summaries each 100 steps
            if self.model.optimizer.iterations % 100 == 0:
                tf.summary.experimental.set_step(self.model.optimizer.iterations)
                with self._writer.as_default():
                    for name, value in zip(self.model.metrics_names, metrics):
                        tf.summary.scalar("train/{}".format(name), value)

    def evaluate(self, dataset, dataset_name, args):
        # We assume that model metric are already resetted at this point.
        self.model.reset_metrics()
        for batch in dataset.batches(args.batch_size):
            # TODO: Evaluate the given batch with `test_on_batch`, using the
            # same inputs as in training, but pass `reset_metrics=False` to
            # aggregate the metrics. Store the metrics of the last batch as `metrics`.

            # word_ids = tf.keras.preprocessing.sequence.pad_sequences(batch[dataset.FORMS].word_ids, padding='post')
            # charseqs = tf.keras.preprocessing.sequence.pad_sequences(batch[dataset.FORMS].charseqs, padding='post')
            #metrics = self.model.test_on_batch([word_ids, charseqs],
            metrics = self.model.test_on_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseqs],
                                               batch[dataset.TAGS].word_ids,
                                               reset_metrics=False)

        metrics = dict(zip(self.model.metrics_names, metrics))
        with self._writer.as_default():
            tf.summary.experimental.set_step(self.model.optimizer.iterations)
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        return metrics


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
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

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args,
                      num_words=len(morpho.train.data[morpho.train.FORMS].words),
                      num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                      num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    for epoch in range(args.epochs):
        network.train_epoch(morpho.train, args)
        metrics = network.evaluate(morpho.dev, "dev", args)

    metrics = network.evaluate(morpho.test, "test", args)
    with open("tagger_we.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["accuracy"]), file=out_file)
