#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, pdt, args):

        num_words = len(pdt.train.data[morpho.train.FORMS].words)
        num_tags = len(pdt.train.data[morpho.train.TAGS].words)
        num_chars = len(pdt.train.data[morpho.train.FORMS].alphabet)

        word_ids = tf.keras.layers.Input(shape=[None])
        wle = tf.keras.layers.Embedding(num_words, args.we_dim, mask_zero=True)(word_ids)
        charseqs = tf.keras.layers.Input(shape=[None, None])
        valid_words = tf.where(word_ids != 0)

        cle = self.cle_embedding(charseqs, valid_words, num_chars, args)

        hidden = tf.keras.layers.Concatenate()([wle, cle])

        args.rnn_cell_dim = args.we_dim + args.cle_dim * 2

        hidden = self.bidirectional_rnn(hidden, args)

        predictions = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_tags, activation='softmax'))(hidden)

        self.model = tf.keras.Model(inputs=[word_ids, charseqs], outputs=predictions)
        self.model.compile(optimizer=self.get_optimizer(args),
                           loss=tf.losses.SparseCategoricalCrossentropy(),
                           metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def cle_embedding(self, charseqs, valid_words, num_chars, args):
        cle = tf.gather_nd(charseqs, valid_words)
        cle = tf.keras.layers.Embedding(num_chars, args.cle_dim, mask_zero=True)(cle)

        cle = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(args.cle_dim, return_sequences=False, name="CL_GRU"), merge_mode='concat')(cle)

        cle = tf.scatter_nd(valid_words, cle, [tf.shape(charseqs)[0], tf.shape(charseqs)[1], cle.shape[-1]])
        return cle

    def bidirectional_rnn(self, hidden, args):
        residual = hidden
        for _ in range(args.num_layers):
            if args.rnn_cell == 'LSTM':
                hidden = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True, name="LSTM"), merge_mode='sum')(hidden)
            elif args.rnn_cell == 'GRU':
                hidden = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(args.rnn_cell_dim, return_sequences=True, name="GRU"), merge_mode='sum')(hidden)
            else:
                hidden = tf.keras.layers.Bidirectional(
                    tf.keras.layers.SimpleRNN(args.rnn_cell_dim, return_sequences=True, name="GRU"), merge_mode='sum')(hidden)

            hidden += residual

        return hidden

    def train(self, pdt, args):
        for idx in range(args.epochs):
            for train_batch in tqdm(pdt.train.batches(args.batch_size), desc="Epoch {}".format(idx)):
                self.train_batch(train_batch)

            self.evaluate(pdt.dev, 'validation', args)

    def train_batch(self, batch):
        metrics = self.model.train_on_batch([batch[MorphoDataset.Dataset.FORMS].word_ids,
                                             batch[MorphoDataset.Dataset.FORMS].charseqs],
                                            batch[MorphoDataset.Dataset.TAGS].word_ids,
                                            reset_metrics=True)


        # Generate the summaries each 100 steps
        if self.model.optimizer.iterations % 100 == 0:
            tf.summary.experimental.set_step(self.model.optimizer.iterations)
            with self._writer.as_default():
                for name, value in zip(self.model.metrics_names, metrics):
                    tf.summary.scalar("train/{}".format(name), value)

    def evaluate(self, dataset, dataset_name, args):
        self.model.reset_metrics()
        for batch in tqdm(dataset.batches(args.batch_size), desc="Evaluation on {}".format(dataset_name)):
            metrics = self.model.test_on_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseqs],
                                               batch[dataset.TAGS].word_ids,
                                               reset_metrics=False)

        metrics = dict(zip(self.model.metrics_names, metrics))
        with self._writer.as_default():
            tf.summary.experimental.set_step(self.model.optimizer.iterations)
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)
                print("{} {}: {}".format(dataset_name, name, value))

    def predict(self, dataset, args):
        predictions = []  # self.model.predict([dataset[dataset.FORMS].word_ids, dataset[dataset.FORMS].charseqs],batch_size=args.batch_size)
        for i, batch in enumerate(tqdm(dataset.batches(args.batch_size), desc="Predicting!")):
            prediction = self.model.predict([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseqs])
            prediction = prediction.argmax(axis=-1)
            predictions += list(prediction)

        print(predictions)
        return (predictions)

    def test(self, pdt, args):
        out_path = "tagger_competition_test.txt"
        if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
        with open(out_path, "w", encoding="utf-8") as out_file:
            for i, sentence in enumerate(self.predict(pdt.test, args)):
                for j in range(len(pdt.test.data[pdt.test.FORMS].word_strings[i])):
                    print(pdt.test.data[pdt.test.FORMS].word_strings[i][j],
                          pdt.test.data[pdt.test.LEMMAS].word_strings[i][j],
                          pdt.test.data[pdt.test.TAGS].words[sentence[j]],
                          sep="\t", file=out_file)
                print(file=out_file)


    def get_optimizer(self, args):
        learning_rate_final = args.learning_rate_final
        decay_steps = int(90828 * args.epochs / args.batch_size)
        if args.decay == 'polynomial':
            learning_rate_schedule = tf.optimizers.schedules.PolynomialDecay(args.learning_rate,
                                                                             decay_steps=decay_steps,
                                                                             end_learning_rate=args.learning_rate_final)
        elif args.decay == 'exponential':
            decay_rate = learning_rate_final / args.learning_rate
            learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                                              decay_steps=decay_steps,
                                                                              decay_rate=decay_rate, staircase=False)
        elif args.decay == "onplateau":
            learning_rate_schedule = args.learning_rate
            self._callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                                                        min_lr=args.learning_rate_final))

        else:
            learning_rate_schedule = args.learning_rate

        optimizer = None
        if args.optimizer == 'SGD':
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate_schedule, momentum=args.momentum)
        elif args.optimizer == "RMSProp":
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate_schedule, momentum=args.momentum)
        elif args.optimizer == 'Adam':
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule)

        return optimizer

    def save(self, curr_date, args):
        self.model.save(os.path.join(args.logdir, "{}-{:.4f}-model.h5".format(curr_date, 0.00), include_optimizer=False))


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument("--batch-size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # RNN architecture
    parser.add_argument("--num-layers", default=2, type=int, help="Number of RNN layers")
    parser.add_argument("--rnn-cell-dim", default=512, type=int, help="Dimensionality of RNN latent vector")
    parser.add_argument("--rnn-cell", default='LSTM', type=str, help='Type of RNN cell (LSTM, GRU, or SimpleRNN')
    # Embeddings
    parser.add_argument("--we-dim", default=128, type=int, help="Dimensionality of word embeddings")
    parser.add_argument("--cle-dim", default=64, type=int, help="Dimensionality of character level embeddings")
    parser.add_argument("--rnn-cle", action='store_true', help="Whether to use RNN chracter level embeddings")
    # Optimizer parameters
    parser.add_argument("--optimizer", default='Adam', type=str, help="NN optimizer")
    parser.add_argument("--momentum", default=0., type=float, help="Momentum of gradient schedule.")
    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--learning-rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning-rate-final", default=None, type=float, help="Final learning rate.")

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
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        curr_date,
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Create the network and train
    network = Network(morpho, args)
    network.train(morpho, args)
    network.test(morpho,args)

    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.

