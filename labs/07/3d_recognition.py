#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from modelnet import ModelNet

import efficient_net3D


# this is needed to load Efficient NET keras model
def swish_activation(x):
    return (tf.keras.activations.sigmoid(x) * x)

tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish_activation)})


class Network:
    def __init__(self, args):

        self._model = efficient_net3D.EfficientNetB0(True, resolution=args.modelnet,
                                                     drop_connect=args.drop_connect,
                                                     dropout_rate=args.dropout,
                                                     weights=None,
                                                     classes = len(ModelNet.LABELS))

        self._callbacks = []
        self._optimizer = self.get_optimizer(args)

        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        )

        self._callbacks.append(tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1,
                                                                update_freq=100, profile_batch=0))
        self._callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
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

    def train(self, mnet, args):

        train = tf.data.Dataset.from_tensor_slices(mnet.train.data)

        train = train.map(lambda x: (x["voxels"], tf.one_hot(x["labels"], depth=len(ModelNet.LABELS))))
        train = train.batch(args.batch_size).prefetch(args.threads)


        dev = tf.data.Dataset.from_tensor_slices(mnet.dev.data)
        dev = dev.map(lambda x: (x["voxels"], tf.one_hot(x["labels"], depth=len(ModelNet.LABELS))))
        dev = dev.batch(args.batch_size).prefetch(args.threads)

        self.model_history = self._model.fit(train,
                                             epochs=args.epochs,
                                             validation_data=dev,
                                             callbacks=self._callbacks)

    def test(self, mnet, args):
        test = tf.data.Dataset.from_tensor_slices(mnet.test.data)
        test = test.map(lambda x: x["voxels"])
        test = test.batch(args.batch_size)
        with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as out_file:
            for probs in self._model.predict(test):
                print(np.argmax(probs), file=out_file)

    def save(self, args, curr_date):
        self._model.save(os.path.join(args.logdir, "{}-{:.4f}-model.h5".
                                      format(curr_date, max(self.model_history.history['val_accuracy'][-20:]))), include_optimizer=False)

    # @staticmethod
    # def train_augment(image, label, cut_out=16):
    #     if tf.random.uniform([]) >= 0.5:
    #         image = tf.image.flip_left_right(image)
    #     image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 6, CAGS.W + 6)
    #     image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 12, dtype=tf.int32),
    #                                     tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 12, dtype=tf.int32)])
    #     image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])
    #
    #     # random global properties
    #     image = tf.image.random_jpeg_quality(image, 0.9, 1.0)
    #
    #
    #     #cut_out
    #
    #     mask = np.ones((CAGS.H + 2*cut_out, CAGS.W + 2*cut_out, CAGS.C), np.float32)
    #     mean_pixels = np.zeros((CAGS.H + 2*cut_out, CAGS.W + 2*cut_out, CAGS.C), np.float32)
    #     rnd_H = np.random.randint(CAGS.H + cut_out)
    #     rnd_W = np.random.randint(CAGS.W + cut_out)
    #
    #     mask[rnd_H:rnd_H + cut_out, rnd_W + cut_out, :] = 0
    #     mask = tf.constant(mask)
    #     mask = tf.image.resize_with_crop_or_pad(mask, CAGS.H, CAGS.W)
    #
    #     mean_pixels[rnd_H:rnd_H + cut_out, rnd_W + cut_out, :] = np.array([0.15610054, 0.15610054, 0.15610054], np.float32)
    #     mean_pixels = tf.constant(mean_pixels)
    #     mean_pixels = tf.image.resize_with_crop_or_pad(mean_pixels, CAGS.H, CAGS.W)
    #
    #     image = image * mask + mean_pixels
    #     return image, label
    #
    # def efficient_net(self, args):
    #
    #     efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False, drop_connect=args.drop_connect)
    #     num_layers = len(efficientnet_b0.layers)
    #
    #     for lidx, layer in enumerate(efficientnet_b0.layers):
    #         if ('conv' in layer.name or 'expand' in layer.name or 'reduce' in layer.name) \
    #                 and lidx/num_layers >= args.freeze:
    #             layer.kernel_regularizer = tf.keras.regularizers.l2(args.l2)
    #             layer.trainable = True
    #         elif ('drop' in layer.name or 'bg' in layer.name or 'gn' in layer.name):
    #             layer.trainable = True
    #         else:
    #             layer.trainable = False
    #
    #     return efficientnet_b0

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # Regularization parameters
    parser.add_argument("--l2", default=0., type=float, help="L2 regularization.")
    parser.add_argument("--label-smoothing", default=0., type=float, help="Label smoothing.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout in top layer")
    parser.add_argument("--drop-connect", default=0.2, type=float, help="Drop connection probability in efficient net.")
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
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnet = ModelNet(args.modelnet)

    # TODO: Create the model and train it
    model = Network(args)
    model.train(mnet, args)
    model.test(mnet, args)