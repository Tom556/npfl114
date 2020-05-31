#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import transformers
from tqdm import tqdm

from text_classification_dataset import TextClassificationDataset

class Network:
    def __init__(self, args, labels):
        # TODO: Define a suitable model.
        self._model = transformers.TFBertForSequenceClassification.from_pretrained(args.bert,
                                                                                   num_labels=labels,
                                                                                   hidden_dropout_prob=args.dropout,
                                                                                   attention_probs_dropout_prob=args.dropout)

        self._model.layers[0].trainable = False
        self._optimizer = self.get_optimizer(args)
        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)


    def train(self, data, args):
        # TODO: Train the network on a given dataset.
        for eidx in range(args.pre_epochs):
            self.train_epoch(data, args, eidx)
            self.evaluate(data.dev, "dev", args)

        self._model.layers[0].trainable = True
        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )

        for eidx in range(args.pre_epochs, args.epochs):
            self.train_epoch(data, args, eidx)
            self.evaluate(data.dev, "dev", args)

    def train_epoch(self,data, args, epoch_idx):
        for iteration, batch in enumerate(tqdm(data.train.batches(args.batch_size), desc=f"Epoch {epoch_idx} training")):
            batch_tokens, batch_labels = batch
            batch_mask = np.sign(batch_tokens)
            metrics = self._model.train_on_batch([batch_tokens, batch_mask], batch_labels)
            if iteration % 100 == 0:
                with self._writer.as_default():
                    tf.summary.experimental.set_step(self._model.optimizer.iterations)
                    for name, value in zip(self._model.metrics_names, metrics):
                        tf.summary.scalar("train/{}".format(name), value)
                        print(f"{name}: {value}")

    def evaluate(self, dataset, dataset_name, args):
        self._model.reset_metrics()
        iterator = tqdm(enumerate(dataset.batches(args.batch_size)), desc=f"Evaluating on {dataset_name}.")
        for iteration, batch in iterator:
            batch_tokens, batch_labels = batch
            batch_mask = np.sign(batch_tokens)
            metrics = self._model.test_on_batch([batch_tokens, batch_mask], batch_labels, reset_metrics=False)
            iterator.set_description_str(f"Evaluating on {dataset_name}. loss: {metrics[0]}, accuracy: {metrics[1]}")

        with self._writer.as_default():
            tf.summary.experimental.set_step(self._model.optimizer.iterations)
            for name, value in zip(self._model.metrics_names, metrics):
                tf.summary.scalar(f"{dataset_name}/{name}", value)

    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndarray of the
        # predicted label indices (no probabilities/distributions).
        for i, batch in enumerate(tqdm(dataset.batches(args.batch_size), desc="Predicting!")):
            batch_tokens, batch_labels = batch
            batch_mask = np.sign(batch_tokens)
            batch_predictions = self._model.predict_on_batch([batch_tokens, batch_mask])
            for prediction in batch_predictions[0]:
                yield np.argmax(prediction)

    def get_optimizer(self, args):
        learning_rate_final = args.learning_rate_final
        decay_steps = int(7752 * args.epochs / args.batch_size)
        if args.decay == 'polynomial':
            self.learning_rate_schedule = tf.optimizers.schedules.PolynomialDecay(args.learning_rate,
                                                                             decay_steps=decay_steps,
                                                                             end_learning_rate=args.learning_rate_final)
        elif args.decay == 'exponential':
            decay_rate = learning_rate_final / args.learning_rate
            self.learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                                              decay_steps=decay_steps,
                                                                              decay_rate=decay_rate, staircase=False)

        else:
            self.learning_rate_schedule = args.learning_rate

        optimizer = None
        if args.optimizer == 'SGD':
            optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate_schedule, momentum=args.momentum)
        elif args.optimizer == "RMSProp":
            optimizer = tf.optimizers.RMSprop(learning_rate=self.learning_rate_schedule, momentum=args.momentum)
        elif args.optimizer == 'Adam':
            optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate_schedule)

        return optimizer


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size.")
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="BERT model.")
    parser.add_argument("--pre-epochs", default=3, type=int, help="Number of pretraining epochs")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # training arguments
    parser.add_argument("--optimizer", default='Adam', type=str, help="NN optimizer")
    parser.add_argument("--momentum", default=0., type=float, help="Momentum of gradient schedule.")
    parser.add_argument("--decay", default="exponential", type=str, help="Learning decay rate type")
    parser.add_argument("--learning-rate", default=1e-4, type=float, help="Initial learning rate.")
    parser.add_argument("--learning-rate-final", default=1e-8, type=float, help="Final learning rate.")
    #
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")

    args = parser.parse_args([] if "__file__" not in globals() else None)
    #args.epochs = [(int(epochs), float(lr)) for epochslr in args.epochs.split(",") for epochs, lr in [epochslr.split(":")]]

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

    # TODO: Create the BERT tokenizer to `tokenizer` variable

    tokenizer = transformers.BertTokenizer.from_pretrained(args.bert, do_lower_case=True)

    # TODO: Load the data, using a correct `tokenizer` argument, which
    # should be a callable that given a sentence in a string produces
    # a list/np.ndarray of token integers.
    facebook = TextClassificationDataset("czech_facebook", tokenizer=tokenizer)

    # Create the network and train
    network = Network(args, len(facebook.train.LABELS))
    network.train(facebook, args)
    network.evaluate(facebook.test, "test", args)

    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.
    out_path = "sentiment_analysis_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for label in network.predict(facebook.test, args):
            print(facebook.test.LABELS[label], file=out_file)
