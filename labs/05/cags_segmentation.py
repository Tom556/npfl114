#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
from cags_segmentation_eval import CAGSMaskIoU
import efficient_net

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

class Network:
    def __init__(self, args):

        base_model = self.efficient_net(args)

        output = self.up_scale(base_model.input, base_model.output, args)

        self._model = tf.keras.Model(inputs=base_model.input, outputs=output)

        self._callbacks = []
        self._optimizer = self.get_optimizer(args)

        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
            metrics=[CAGSMaskIoU(name="iou")],
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
        train = train.map(lambda x: (x["image"], x["mask"]))
        train = train.shuffle(5000, seed=args.seed)
        # train = train.map(self.train_augment)
        train = train.batch(args.batch_size).prefetch(args.threads)

        dev = cags.dev
        dev = dev.map(CAGS.parse)
        dev = dev.map(lambda x: (x["image"], x["mask"]))
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
        with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as out_file:
            test_masks = self._model.predict(test)
            for mask in test_masks:
                zeros, ones, runs = 0, 0, []
                for pixel in np.reshape(mask >= 0.5, [-1]):
                    if pixel:
                        if zeros or (not zeros and not ones):
                            runs.append(zeros)
                            zeros = 0
                        ones += 1
                    else:
                        if ones:
                            runs.append(ones)
                            ones = 0
                        zeros += 1
                runs.append(zeros + ones)
                print(*runs, file=out_file)

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
            if ('conv' in layer.name or 'expand' in layer.name or 'reduce' in layer.name) \
                    and lidx/num_layers >= args.freeze:
                layer.kernel_regularizer = tf.keras.regularizers.l2(args.l2)
                layer.trainable = True
            elif ('drop' in layer.name or 'bg' in layer.name or 'gn' in layer.name):
                layer.trainable = True
            else:
                layer.trainable = False

        return efficientnet_b0

    def up_scale(self, input, down_scaled_output, args):

        x = down_scaled_output[1]
        for idx, dso in enumerate(down_scaled_output[2:] + [input]):
            up_filters = dso.shape[-1]
            x = tf.keras.layers.Conv2DTranspose(filters=up_filters,
                                                kernel_size=1,
                                                padding='same',
                                                strides=2,
                                                kernel_initializer=CONV_KERNEL_INITIALIZER,
                                                kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                                name ='upscale_' + str(idx))(x)
            x = x + dso
            x = tf.keras.layers.Conv2D(up_filters, 3,
                                       strides=1,
                                       padding='same',
                                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                                       kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                       name='up_conv_' + str(idx) + '_a')(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, name='up_bn_' + str(idx) + '_a')(x)
            x = tf.keras.layers.Activation(tf.nn.swish, name='up_activation_' + str(idx) + '_a')(x)

            x = tf.keras.layers.Conv2D(up_filters, 3,
                                       strides=1,
                                       padding='same',
                                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                                       kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                       name='up_conv_' + str(idx) + '_b')(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, name='up_bn_' + str(idx) + '_b')(x)
            x = tf.keras.layers.Activation(tf.nn.swish, name='up_activation_' + str(idx) + '_b')(x)

        x = tf.keras.layers.Conv2D(1, 1,
                                   strides=1,
                                   padding='same',
                                   kernel_initializer=CONV_KERNEL_INITIALIZER,
                                   kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                   name='up_top_conv')(x)
        output = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name='up_top_activation')(x)
        return output




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

    # Create logdir name
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        curr_date,
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    # Load the data
    cags = CAGS()

    cags_network = Network(args)
    cags_network.train(cags, args)
    cags_network.test(cags, args)
    cags_network.save(args, curr_date)
