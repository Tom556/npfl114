#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

class Network:

    ES_DELTA = 1e-4
    ES_PATIENCE = 5
    OP_PATIENCE = 1
    OP_REDUCE_LR = 0.1
    class Lemmatizer(tf.keras.Model):
        def __init__(self, args, num_source_chars, num_target_chars):
            super().__init__()
            self.source_embedding = tf.keras.layers.Embedding(num_source_chars, args.cle_dim, mask_zero=True,
                                                              name='encoder_embedding')

            self.source_rnn = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(args.rnn_dim, return_sequences=True), merge_mode='sum', name='encoder_rnn')

            self.target_embedding = tf.keras.layers.Embedding(num_source_chars, args.cle_dim, mask_zero=False,
                                                              name='target_embedding')

            self.target_rnn_cell = tf.keras.layers.GRUCell(args.rnn_dim)

            self.target_output_layer = tf.keras.layers.Dense(num_target_chars, name='target_output')

            self.attention_source_layer = tf.keras.layers.Dense(args.rnn_dim)
            self.attention_state_layer = tf.keras.layers.Dense(args.rnn_dim)
            self.attention_weight_layer = tf.keras.layers.Dense(1)

            self.num_logits = num_target_chars

        class DecoderTraining(tfa.seq2seq.BaseDecoder):
            def __init__(self, lemmatizer, *args, **kwargs):
                self.lemmatizer = lemmatizer
                super().__init__.__wrapped__(self, *args, **kwargs)

            @property
            def batch_size(self):
                return tf.shape(self.source_states)[0]

            @property
            def output_size(self):
                return tf.TensorShape([self.lemmatizer.num_logits])
            @property
            def output_dtype(self):
                return tf.float32

            def _with_attention(self, inputs, states):

                query_attentions = tf.expand_dims(self.lemmatizer.attention_state_layer(states), 1)
                weight_attention = self.lemmatizer.attention_weight_layer(
                    tf.keras.activations.tanh(self.key_attention + query_attentions))

                weight_attention = tf.keras.activations.softmax(weight_attention, axis=1)
                attention = tf.reduce_sum(weight_attention * self.source_states, axis=1)
                return tf.keras.layers.Concatenate(axis=-1)([inputs, attention])

            def initialize(self, layer_inputs, initial_state=None, mask=None):
                self.source_states, self.targets = layer_inputs

                finished = tf.fill([self.batch_size], False)
                inputs = self.lemmatizer.target_embedding(tf.fill([self.batch_size], MorphoDataset.Factor.BOW))
                states = self.source_states[:,0,:]

                self.key_attention = self.lemmatizer.attention_source_layer(self.source_states)
                inputs = self._with_attention(inputs, states)
                return finished, inputs, states

            def step(self, time, inputs, states, training):
                outputs, [states] = self.lemmatizer.target_rnn_cell(inputs, [states], training)
                outputs = self.lemmatizer.target_output_layer(outputs)
                next_inputs = self.lemmatizer.target_embedding(self.targets[:,time])
                finished = (self.targets[:,time] == MorphoDataset.Factor.EOW)

                next_inputs = self._with_attention(next_inputs, states)

                return outputs, states, next_inputs, finished

        class DecoderPrediction(DecoderTraining):
            @property
            def output_size(self):
                 return tf.TensorShape([])
            @property
            def output_dtype(self):
                 return tf.int32

            def initialize(self, layer_inputs, initial_state=None, mask=None):
                self.source_states = layer_inputs
                finished = tf.fill([self.batch_size], False)
                inputs = self.lemmatizer.target_embedding(tf.fill([self.batch_size], MorphoDataset.Factor.BOW))
                states = self.source_states[:,0,:]

                self.key_attention = self.lemmatizer.attention_source_layer(self.source_states)
                inputs = self._with_attention(inputs, states)
                return finished, inputs, states

            def step(self, time, inputs, states, training):

                outputs, [states] = self.lemmatizer.target_rnn_cell(inputs, [states], training)
                outputs = self.lemmatizer.target_output_layer(outputs)
                outputs = tf.argmax(outputs, axis=-1, output_type=tf.int32)
                next_inputs = self.lemmatizer.target_embedding(outputs)
                finished = (outputs == MorphoDataset.Factor.EOW)

                next_inputs = self._with_attention(next_inputs, states)

                return outputs, states, next_inputs, finished

        def call(self, inputs):
            if isinstance(inputs, list) and len(inputs) == 2:
                source_charseqs, target_charseqs = inputs
            else:
                source_charseqs, target_charseqs = inputs, None
            source_charseqs_shape = tf.shape(source_charseqs)

            valid_words = tf.cast(tf.where(source_charseqs[:, :, 0] != 0), tf.int32)
            source_charseqs = tf.gather_nd(source_charseqs, valid_words)
            if target_charseqs is not None:
                target_charseqs = tf.gather_nd(target_charseqs, valid_words)

            source = self.source_embedding(source_charseqs)
            source_states = self.source_rnn(source)
            # Run the appropriate decoder
            if target_charseqs is not None:
                decoder_training = self.DecoderTraining(self)
                output_layer, _, output_lens = decoder_training([source_states, target_charseqs])
            else:

                decoder_predict = self.DecoderPrediction(self, maximum_iterations=(tf.shape(source_charseqs)[1]+10))
                output_layer, _ , output_lens = decoder_predict(source_states)

            output_layer = tf.scatter_nd(valid_words, output_layer, tf.concat([source_charseqs_shape[:2], tf.shape(output_layer)[1:]], axis=0))
            output_layer._keras_mask = tf.sequence_mask(tf.scatter_nd(valid_words, output_lens, source_charseqs_shape[:2]))
            return output_layer

    def __init__(self, args, num_source_chars, num_target_chars):
        self.lemmatizer = self.Lemmatizer(args, num_source_chars, num_target_chars)

        self._optimizer = self.get_optimizer(args)
        self.optimal_metric = 0.

        self.lemmatizer.compile(
            optimizer=self._optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="character_accuracy")]
        )
        self.writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def get_optimizer(self, args):
        learning_rate_final = args.learning_rate_final
        decay_steps = int(90828 * args.epochs / args.batch_size)
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
            optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate_schedule, momentum=args.momentum, clipnorm=args.clipnorm)
        elif args.optimizer == "RMSProp":
            optimizer = tf.optimizers.RMSprop(learning_rate=self.learning_rate_schedule, momentum=args.momentum, clipnorm=args.clipnorm)
        elif args.optimizer == 'Adam':
            optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate_schedule, clipnorm=args.clipnorm)

        return optimizer

    def append_eow(self, sequences):
        """Append EOW character after end of every given sequence."""
        padded_sequences = np.pad(sequences, [[0, 0], [0, 0], [0, 1]])
        ends = np.logical_xor(padded_sequences != 0, np.pad(sequences, [[0, 0], [0, 0], [1, 0]], constant_values=1) != 0)
        padded_sequences[ends] = MorphoDataset.Factor.EOW
        return padded_sequences

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):

            targets = self.append_eow(batch[dataset.LEMMAS].charseqs)

            metrics = self.lemmatizer.train_on_batch([batch[dataset.FORMS].charseqs, targets], targets)

            # Generate the summaries each 10 steps
            iteration = int(self.lemmatizer.optimizer.iterations)
            if iteration % 100 == 0:
                tf.summary.experimental.set_step(iteration)
                metrics = dict(zip(self.lemmatizer.metrics_names, metrics))

                predictions = self.predict_batch(batch[dataset.FORMS].charseqs[:1]).numpy()
                form = "".join(dataset.data[dataset.FORMS].alphabet[i] for i in batch[dataset.FORMS].charseqs[0, 0] if i)
                gold_lemma = "".join(dataset.data[dataset.LEMMAS].alphabet[i] for i in targets[0, 0] if i)
                system_lemma = "".join(dataset.data[dataset.LEMMAS].alphabet[i] for i in predictions[0, 0] if i != MorphoDataset.Factor.EOW)
                status = ", ".join([*["{}={:.4f}".format(name, value) for name, value in metrics.items()],
                                    "{} {} {}".format(form, gold_lemma, system_lemma)])
                print("Step {}:".format(iteration), status)

                with self.writer.as_default():
                    for name, value in metrics.items():
                        tf.summary.scalar("train/{}".format(name), value)
                    tf.summary.text("train/prediction", status)

    @tf.function(experimental_relax_shapes=True)
    def predict_batch(self, charseqs):
        return self.lemmatizer(charseqs)

    def evaluate(self, dataset, dataset_name, args):
        correct_lemmas, total_lemmas = 0, 0
        for batch in dataset.batches(args.batch_size):
            predictions = self.predict_batch(batch[dataset.FORMS].charseqs).numpy()

            # Compute whole lemma accuracy
            targets = self.append_eow(batch[dataset.LEMMAS].charseqs)
            resized_predictions = np.concatenate([predictions, np.zeros_like(targets)], axis=2)[:, :, :targets.shape[2]]
            valid_lemmas = targets[:, :, 0] != MorphoDataset.Factor.EOW

            total_lemmas += np.sum(valid_lemmas)
            correct_lemmas += np.sum(valid_lemmas * np.all(targets == resized_predictions * (targets != 0), axis=2))

        metrics = {"lemma_accuracy": correct_lemmas / total_lemmas}
        with self.writer.as_default():
            tf.summary.experimental.set_step(self.lemmatizer.optimizer.iterations)
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        return metrics

    def predict(self, dataset, args):
        for i, batch in enumerate(tqdm(dataset.batches(args.batch_size), desc="Predicting!")):
            batch_predictions = self.predict_batch(batch[dataset.FORMS].charseqs).numpy()
            for prediction in batch_predictions:
                yield prediction

    def training(self, morpho, args):
        curr_patience = 0
        best_weights = None
        for epoch in range(args.epochs):
            self.train_epoch(morpho.train, args)
            metrics = self.evaluate(morpho.dev, "dev", args)
            print("Evaluation on {}, epoch {}: {}".format("dev", epoch + 1, metrics))

            eval_metric = metrics["lemma_accuracy"]
            if eval_metric > self.optimal_metric + self.ES_DELTA:
                self.optimal_metric = eval_metric
                best_weights = self.lemmatizer.get_weights()
                curr_patience = 0
            else:
                curr_patience += 1

            if curr_patience > 0 and args.decay == 'onplateau':
                self.learning_rate_schedule *= self.OP_REDUCE_LR
                self._optimizer.learning_rate.assign(self.learning_rate_schedule)
                # reset_variables = [np.zeros_like(var.numpy()) for var in self._optimizer.variables()]
                # self._optimizer.set_weights(reset_variables)
            if curr_patience > self.ES_PATIENCE:
                self.lemmatizer.set_weights(best_weights)
                break

    def predicting(self, dataset, dataset_name, args):
        # Generate test set annotations, but to allow parallel execution, create it
        # in in args.logdir if it exists.
        out_path = "lemmatizer_competition_{}.txt".format(dataset_name)
        if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
        with open(out_path, "w", encoding="utf-8") as out_file:
            for i, sentence in enumerate(self.predict(dataset, args)):
                for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
                    lemma = []
                    for c in map(int, sentence[j]):
                        if c == MorphoDataset.Factor.EOW: break
                        lemma.append(morpho.test.data[morpho.test.LEMMAS].alphabet[c])

                    print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
                          "".join(lemma),
                          morpho.test.data[morpho.test.TAGS].word_strings[i][j],
                          sep="\t", file=out_file)
                print(file=out_file)

    def save(self, curr_date, args):
        self.lemmatizer.save(os.path.join(args.logdir, "{}-{:.4f}-model.h5".format(curr_date, self.optimal_metric)),
                             include_optimizer=False)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--max-sentences", default=5000, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # Network architecture
    parser.add_argument("--cle-dim", default=64, type=int, help="CLE embedding dimension.")
    parser.add_argument("--rnn-dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--num-layers", default=1, type=int, help="Number of layers in RNN.")
    # Optimizer parameters
    parser.add_argument("--optimizer", default='Adam', type=str, help="NN optimizer")
    parser.add_argument("--momentum", default=0., type=float, help="Momentum of gradient schedule.")
    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--learning-rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning-rate-final", default=1e-8, type=float, help="Final learning rate.")
    # Reguralization
    parser.add_argument("--clip-norm", default=5., type=float, help="Gradient norm clipping value.")
    #parser.add_argument("--dropout", default=0., type=float, help="Dropout in the network.")

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
    network = Network(args,
                      num_source_chars=len(morpho.train.data[morpho.train.FORMS].alphabet),
                      num_target_chars=len(morpho.train.data[morpho.train.LEMMAS].alphabet))
    network.training(morpho, args)
    network.predicting(morpho.dev, 'dev', args)
    network.predicting(morpho.test, 'test', args)
    network.save(curr_date, args)

