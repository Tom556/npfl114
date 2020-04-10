#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net


class Network:
    def __init__(self, args):

        if args.efficient_net:
            base_model = self.efficient_net(args)
        else:
            base_model = tf.keras.Sequential([tf.keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])])

        top_model = tf.keras.models.Sequential()
        # top_model.add(tf.keras.layers.GlobalAveragePooling2D())
        top_model.add(tf.keras.layers.Dropout(args.dropout))
        top_model.add(tf.keras.layers.Dense(len(CAGS.LABELS), activation=tf.nn.softmax,
                                        kernel_regularizer=tf.keras.regularizers.l2(args.l2)))

        self._model = tf.keras.Model(inputs=base_model.input, outputs=top_model(base_model.output[0]))

        self._callbacks = []
        self._optimizer = self.get_optimizer(args)

        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        )

        self._callbacks.append(tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1,
                                                                update_freq=100, profile_batch=0))
        self._callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                                restore_best_weights=True))

    def get_optimizer(self, args):
        learning_rate_final = args.learning_rate_final
        decay_steps = int(2142 * args.epochs / args.batch_size)
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

    def train(self, cags, args):

        train = cags.train
        train = train.map(CAGS.parse)
        train = train.map(lambda x: (x["image"], tf.one_hot(x["label"], depth=len(CAGS.LABELS))))
        train = train.shuffle(5000, seed=args.seed)
        train = train.map(self.train_augment)
        train = train.batch(args.batch_size).prefetch(args.threads)

        dev = cags.dev
        dev = dev.map(CAGS.parse)
        dev = dev.map(lambda x: (x["image"], tf.one_hot(x["label"], depth=len(CAGS.LABELS))))
        dev = dev.batch(args.batch_size).prefetch(args.threads)

        self.model_history = self._model.fit(train,
                                             epochs=args.epochs,
                                             validation_data=dev,
                                             callbacks=self._callbacks)

    def test(self, cags, args):
        test = cags.test
        test = test.map(CAGS.parse)
        test = test.map(lambda x: x["image"])
        test = test.batch(args.batch_size)
        with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as out_file:
            for probs in self._model.predict(test):
                print(np.argmax(probs), file=out_file)

    def save(self, args, curr_date):
        self._model.save(os.path.join(args.logdir, "{}-{:.4f}-model.h5".
                                      format(curr_date, max(self.model_history.history['val_accuracy'][-10:]))), include_optimizer=False)

    @staticmethod
    def train_augment(image, label, cut_out=16):
        if tf.random.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 6, CAGS.W + 6)
        image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 12, dtype=tf.int32),
                                        tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 12, dtype=tf.int32)])
        image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])

        #cut_out

        mask = np.ones((CAGS.H + 2*cut_out, CAGS.W + 2*cut_out, CAGS.C), np.float32)
        mean_pixels = np.zeros((CAGS.H + 2*cut_out, CAGS.W + 2*cut_out, CAGS.C), np.float32)
        #image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 2*cut_out, CAGS.W + 2*cut_out)
        rnd_H = np.random.randint(CAGS.H + cut_out)
        rnd_W = np.random.randint(CAGS.W + cut_out)

        mask[rnd_H:rnd_H + cut_out, rnd_W + cut_out, :] = 0
        mask = tf.constant(mask)
        mask = tf.image.resize_with_crop_or_pad(mask, CAGS.H, CAGS.W)

        mean_pixels[rnd_H:rnd_H + cut_out, rnd_W + cut_out, :] = np.array([0.15610054, 0.15610054, 0.15610054], np.float32)
        mean_pixels = tf.constant(mean_pixels)
        mean_pixels = tf.image.resize_with_crop_or_pad(mean_pixels, CAGS.H, CAGS.W)

        image = image * mask + mean_pixels
        return image, label

    def efficient_net(self, args):

        efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
        num_layers = len(efficientnet_b0.layers)

        for lidx, layer in enumerate(efficientnet_b0.layers):
            if ('conv' in layer.name or 'excite' in layer.name or 'reduce' in layer.name) \
                    and lidx/num_layers >= args.freeze:
                layer.kernel_regularizer = tf.keras.regularizers.l2(args.l2)
                layer.trainable = True
            elif ('drop' in layer.name or 'bg' in layer.name or 'gn' in layer.name):
                layer.trainable = True
            else:
                layer.trainable = False

        return efficientnet_b0



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", default=None, type=str, help="Use ensemble of pre-trained models.")
    # Basic training arguments
    parser.add_argument("--batch-size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # Regularization parameters
    parser.add_argument("--l2", default=0., type=float, help="L2 regularization.")
    parser.add_argument("--label-smoothing", default=0., type=float, help="Label smoothing.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout in top layer")
    # Optimizer parameters
    parser.add_argument("--optimizer", default='Adam', type=str, help="NN optimizer")
    parser.add_argument("--momentum", default=0., type=float, help="Momentum of gradient schedule.")
    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--learning-rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning-rate-final", default=None, type=float, help="Final learning rate.")
    # Transfer learning with resnet
    parser.add_argument("--efficient-net", action="store_true", help="Use EfficientNet network.")
    parser.add_argument("--freeze", default=0., type=float, help="Percentage of frozen layers.")

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

    # Create logdir
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        curr_date,
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()

    # CAGS_TABLE = tf.lookup.StaticVocabularyTable(
    #     tf.lookup.KeyValueTensorInitializer(CAGS.LABELS, tf.range(len(CAGS.LABELS), dtype=tf.int64)), 1)

    cifar_network = Network(args)
    cifar_network.train(cags, args)
    cifar_network.test(cags, args)
    cifar_network.save(args, curr_date)
