#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from timit_mfcc import TimitMFCC

class Network:

    ES_DELTA = 1e-4
    ES_PATIENCE = 4
    TRAIN_EXAMPLES = 3512
    def __init__(self, args):
        self._beam_width = args.ctc_beam

        # TODO: Define a suitable model, given already masked `mfcc` with shape
        # `[None, TimitMFCC.MFCC_DIM]`. The last layer should be a Dense layer
        # without an activation and with `len(TimitMFCC.LETTERS) + 1` outputs,
        # where the `+ 1` is for the CTC blank symbol.
        #
        # Store the results in `self.model`.

        mfcc_signal = tf.keras.layers.Input([None, TimitMFCC.MFCC_DIM])
        hidden = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(args.rnn_cell_dim, activation='relu'))(mfcc_signal)
        hidden = self.bidirectional_rnn(hidden, args)
        output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(len(TimitMFCC.LETTERS) + 1,
                                  activation='softmax',
                                  kernel_regularizer=tf.keras.regularizers.l2(args.l2)))(hidden)
        self.model = tf.keras.Model(inputs=mfcc_signal, outputs=output)

        # The following are just defaults, do not hesitate to modify them.

        self.optimal_loss = np.inf
        self._optimizer = self.get_optimizer(args)
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._metrics = {"loss": tf.metrics.Mean(), "edit_distance": tf.metrics.Mean()}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

        # self.model.compile(optimizer=self._optimizer,
        #                    loss=self._loss,
        #                    metrics=self._metrics)

    def bidirectional_rnn(self, hidden, args):
        for i in range(args.num_layers):
            residual = hidden
            if args.rnn_cell == 'LSTM':
                hidden = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True, name="LSTM",
                                         kernel_regularizer=tf.keras.regularizers.l2(args.l2)),
                    merge_mode='sum')(hidden)
            elif args.rnn_cell == 'GRU':
                hidden = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(args.rnn_cell_dim, return_sequences=True, name="GRU",
                                        kernel_regularizer=tf.keras.regularizers.l2(args.l2)),
                    merge_mode='sum')(hidden)
            else:
                hidden = tf.keras.layers.Bidirectional(
                    tf.keras.layers.SimpleRNN(args.rnn_cell_dim, return_sequences=True, name="GRU",
                                              kernel_regularizer=tf.keras.regularizers.l2(args.l2)),
                    merge_mode='sum')(hidden)
            if i != 0:
                hidden += residual

            hidden = tf.keras.layers.Dropout(args.dropout)(hidden)

        return hidden

    def get_optimizer(self, args):
        learning_rate_final = args.learning_rate_final
        decay_steps = int(self.TRAIN_EXAMPLES * args.epochs / args.batch_size)
        if args.decay == 'polynomial':
            learning_rate_schedule = tf.optimizers.schedules.PolynomialDecay(args.learning_rate,
                                                                             decay_steps=decay_steps,
                                                                             end_learning_rate=args.learning_rate_final)
        elif args.decay == 'exponential':
            decay_rate = learning_rate_final / args.learning_rate
            learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                                              decay_steps=decay_steps,
                                                                              decay_rate=decay_rate, staircase=False)
        # elif args.decay == "onplateau":
        #     learning_rate_schedule = args.learning_rate
        #     self._callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
        #                                                                 min_lr=args.learning_rate_final))

        else:
            learning_rate_schedule = args.learning_rate

        optimizer = None
        if args.optimizer == 'SGD':
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate_schedule, momentum=args.momentum, clipnorm=args.clip_norm)
        elif args.optimizer == "RMSProp":
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate_schedule, momentum=args.momentum, clipnorm=args.clip_norm)
        elif args.optimizer == 'Adam':
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule, clipnorm=args.clip_norm)

        return optimizer

    # Converts given tensor with `0` values for padding elements
    # to a SparseTensor.
    def _to_sparse(self, tensor):
        tensor_indices = tf.where(tf.not_equal(tensor, 0))
        return tf.sparse.SparseTensor(tensor_indices, tf.gather_nd(tensor, tensor_indices), tf.shape(tensor, tf.int64))

    # Convert given sparse tensor to a (dense_output, sequence_lengths).
    def _to_dense(self, tensor):
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor_lens = tf.reduce_sum(tf.cast(tf.not_equal(tensor, -1), tf.int32), axis=1)
        return tensor, tensor_lens

    # Compute logits given input mfcc, mfcc_lens and training flags.
    # Also transpose the logits to `[time_steps, batch, dimension]` shape
    # which is required by the following CTC operations.
    def _compute_logits(self, mfcc, mfcc_lens, training):
        logits = self.model(mfcc, mask=tf.sequence_mask(mfcc_lens), training=training)
        return tf.transpose(logits, [1, 0, 2])

    # Compute CTC loss using given logits, their lengths, and sparse targets.
    def _ctc_loss(self, logits, logits_len, sparse_targets):
        loss = tf.nn.ctc_loss(sparse_targets, logits, None, logits_len, blank_index=len(TimitMFCC.LETTERS))
        self._metrics["loss"](loss)
        sparse_predictions = self._to_sparse(tf.math.argmax(logits, axis=-1, output_type=tf.int32))
        edit_distance = tf.edit_distance(sparse_predictions, sparse_targets, normalize=True)
        self._metrics["edit_distance"](edit_distance)
        return tf.reduce_mean(loss)

    # Perform CTC predictions given logits and their lengths.
    def _ctc_predict(self, logits, logits_len):
        (predictions,), _ = tf.nn.ctc_beam_search_decoder(logits, logits_len, beam_width=self._beam_width)
        return tf.cast(predictions, tf.int32)

    # Compute edit distance given sparse predictions and sparse targets.
    def _edit_distance(self, sparse_predictions, sparse_targets):
        edit_distance = tf.edit_distance(sparse_predictions, sparse_targets, normalize=True)
        self._metrics["edit_distance"](edit_distance)
        return edit_distance

    @tf.function(experimental_relax_shapes=True)
    def train_batch(self, mfcc, mfcc_lens, targets):
        for metric in self._metrics.values():
            metric.reset_states()
        sparse_targets = self._to_sparse(targets)
        with tf.GradientTape() as tape:
            predicted_logits = self._compute_logits(mfcc, mfcc_lens, training=True)
            loss = self._ctc_loss(predicted_logits, mfcc_lens, sparse_targets)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default(), tf.summary.record_if(self._optimizer.iterations % 100 == 0):
            for metric_name, metric in self._metrics.items():
                tf.summary.scalar("train/" + metric_name, metric.result())
                #print(f"{metric_name}: {metric.result().numpy()}")
        return loss

    def train_epoch(self, dataset, args):
        progressbar = tqdm(dataset.batches(args.batch_size))
        for idx, batch in enumerate(progressbar):
            batch_loss = self.train_batch(batch["mfcc"], batch["mfcc_len"], batch["letters"])
            progressbar.set_description(f"Training! loss: {batch_loss}")

    @tf.function(experimental_relax_shapes=True)
    def evaluate_batch(self, mfcc, mfcc_lens, targets):
        sparse_targets = self._to_sparse(targets)
        predicted_logits = self._compute_logits(mfcc, mfcc_lens, training=False)
        loss = self._ctc_loss(predicted_logits, mfcc_lens, sparse_targets)
        return loss

    def evaluate(self, dataset, dataset_name, args):
        for metric in self._metrics.values():
            metric.reset_states()

        progressbar = tqdm(dataset.batches(args.batch_size))
        for idx, batch in enumerate(progressbar):
            batch_loss = self.evaluate_batch(batch["mfcc"], batch["mfcc_len"], batch["letters"])
            progressbar.set_description(f"Evaluating! loss: {batch_loss}")

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for metric_name, metric in self._metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, metric_name), metric.result())
                #print(f"{metric_name}: {metric.result().numpy()}")

        return self._metrics["loss"].result()

    @tf.function(experimental_relax_shapes=True)
    def predict_batch(self, mfcc, mfcc_lens):
        predicted_logits = self._compute_logits(mfcc, mfcc_lens, training=False)
        predicted = self._ctc_predict(predicted_logits, mfcc_lens)
        return self._to_dense(predicted)

    def predict(self, dataset, args):
        sentences = []
        for batch in tqdm(dataset.batches(args.batch_size), desc="Predicting"):
            for prediction, prediction_len in zip(*self.predict_batch(batch["mfcc"], batch["mfcc_len"])):
                sentences.append(prediction[:prediction_len])
        return sentences

    def training(self, args):
        curr_patience = 0
        best_weights = None

        for epoch in range(args.epochs):
            network.train_epoch(timit.train, args)
            eval_loss = network.evaluate(timit.dev, "dev", args)

            if eval_loss < self.optimal_loss - self.ES_DELTA:
                self.optimal_loss = eval_loss
                best_weights = self.model.get_weights()
                curr_patience = 0
            else:
                curr_patience += 1

            if curr_patience > self.ES_PATIENCE:
                self.model.set_weights(best_weights)
                break

    def save(self, curr_date, args):
        self.model.save(os.path.join(args.logdir, "{}-{:.4f}-model.h5".format(curr_date, self.optimal_loss)), include_optimizer=False)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--ctc_beam", default=16, type=int, help="CTC beam.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # RNN architecture
    parser.add_argument("--num-layers", default=2, type=int, help="Number of RNN layers")
    parser.add_argument("--rnn-cell-dim", default=256, type=int, help="Dimensionality of RNN latent vector")
    parser.add_argument("--rnn-cell", default='LSTM', type=str, help='Type of RNN cell (LSTM, GRU, or SimpleRNN')
    # Optimizer parameters
    parser.add_argument("--optimizer", default='Adam', type=str, help="NN optimizer")
    parser.add_argument("--momentum", default=0., type=float, help="Momentum of gradient schedule.")
    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--learning-rate", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--learning-rate-final", default=1e-6, type=float, help="Final learning rate.")
    # Regularization
    parser.add_argument("--l2", default=0., type=float, help="L2 regularization.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout in top layer")
    parser.add_argument('--clip-norm', default=2., type=float, help="Value of l2 norm clipping")

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

    # Load the data
    timit = TimitMFCC()

    # Create the network and train
    network = Network(args)
    network.training(args)
    network.save(curr_date, args)

    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.
    out_path = "speech_recognition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for sentence in network.predict(timit.test, args):
            print(" ".join(timit.LETTERS[letters] for letters in sentence), file=out_file)
