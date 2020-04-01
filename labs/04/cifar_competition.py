#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

class Network:
    def __init__(self, args):

        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        if args.resnet:
            hidden = self.add_resnet(args, inputs)
        else:
            hidden = inputs

        hidden = self.add_cnn(args, args.cnn, hidden)
        # Add the final output layer
        outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self._optimizer = self.get_optimizer(args)

        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        )
        self._callbacks = []
        self._callbacks.append(tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1,
                                                                update_freq=100, profile_batch=0))
        self._callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3))

    def get_optimizer(self, args):
        learning_rate_final = args.learning_rate_final
        decay_steps = int(args.train_size * args.epochs / args.batch_size)
        if args.decay == 'polynomial':
            learning_rate_schedule = tf.optimizers.schedules.PolynomialDecay(args.learning_rate,
                                                                             decay_steps=decay_steps,
                                                                             end_learning_rate=learning_rate_final)
        elif args.decay == 'exponential':
            decay_rate = learning_rate_final / args.learning_rate
            learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                                              decay_steps=decay_steps,
                                                                              decay_rate=decay_rate, staircase=False)
        else:
            learning_rate_schedule = args.learning_rate

        optimizer = None
        if args.optimizer == 'SGD':
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate_schedule)
        elif args.optimizer == "RMSProp":
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate_schedule)
        elif args.optimizer == 'Adam':
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule)

        return optimizer

    def train(self, cifar, args):
        train = tf.data.Dataset.from_tensor_slices((cifar.train.data["images"],
                                                    tf.one_hot(cifar.train.data["labels"], CIFAR10.LABELS)))
        train = train.shuffle(5000, seed=args.seed)
        train = train.map(self.train_augment)
        train = train.batch(args.batch_size)

        dev = tf.data.Dataset.from_tensor_slices((cifar.dev.data["images"],
                                                  tf.one_hot(cifar.dev.data["labels"], CIFAR10.LABELS)))
        dev = dev.batch(args.batch_size)
        self._model.fit(train,
                        epochs=args.epochs,
                        validation_data=dev,
                        callbacks=self._callbacks)

    def test(self, cifar, args):
        test = tf.data.Dataset.from_tensor_slices(cifar.test.data["images"])
        test = test.batch(args.batch_size)
        with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
            for probs in self._model.predict(test):
                print(np.argmax(probs), file=out_file)

    def save(self, args):
        self._model.save(os.path.join(args.logdir, "model.h5"), include_optimizer=False)

    @staticmethod
    def train_augment(image, label):
        if tf.random.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CIFAR10.H + 6, CIFAR10.W + 6)
        image = tf.image.resize(image, [tf.random.uniform([], minval=CIFAR10.H, maxval=CIFAR10.H + 12, dtype=tf.int32),
                                        tf.random.uniform([], minval=CIFAR10.W, maxval=CIFAR10.W + 12, dtype=tf.int32)])
        image = tf.image.random_crop(image, [CIFAR10.H, CIFAR10.W, CIFAR10.C])
        return image, label

    def add_resnet(self, args, inputs):
        if args.resnet == '50':
            resnet = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights=args.resnet_weights,
                                                                input_tensor=inputs)
        elif args.resnet == '101':
            resnet = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights=args.resnet_weights,
                                                                 input_tensor=inputs)
        elif args.resnet == '152':
            resnet = tf.keras.applications.resnet_v2.ResNet152V2(include_top=False, weights=args.resnet_weights)

        # print(len(resnet.layers))
        # for layer in resnet.layers[:-args.num_ft]:
        #     layer.trainable = False

        return resnet.output

    def add_cnn(self,args, parameters, inputs):
        hidden = inputs
        res_blocks = re.findall('\[.*\]', parameters)
        params = re.sub('\-\[.*\]', '', parameters)

        for layer_params in params.split(','):
            layer_params = layer_params.split("-")
            if layer_params[0] in {'C', 'CB'}:
                _, filters, kernel_size, stride, padding = layer_params
                hidden = tf.keras.layers.Conv2D(filters=int(filters), kernel_size=int(kernel_size), strides=int(stride),
                                                padding=padding, use_bias=(layer_params[0] == 'C'),
                                                data_format='channels_last')(hidden)
                if layer_params[0] == 'CB':
                    hidden = tf.keras.layers.BatchNormalization(axis=-1)(hidden)
                hidden = tf.keras.activations.relu(hidden)

            elif layer_params[0] == 'M':
                _, kernel_size, stride = layer_params
                hidden = tf.keras.layers.MaxPool2D(pool_size=int(kernel_size), strides=int(stride),
                                                   data_format='channels_last')(hidden)

            elif layer_params == 'GM':
                hidden = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(hidden)

            elif layer_params[0] == 'GA':
                hidden = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(hidden)

            elif layer_params[0] == 'R':
                hidden += self.add_cnn(args, res_blocks.pop(0)[1:-1], hidden)

            elif layer_params[0] == 'F':
                hidden = tf.keras.layers.Flatten(data_format='channels_last')(hidden)

            elif layer_params[0] == 'H':
                _, hidden_layer_size = layer_params
                hidden = tf.keras.layers.Dense(units=int(hidden_layer_size), activation=tf.nn.relu)(hidden)

            elif layer_params[0] == 'D':
                _, dropout_rate = layer_params
                hidden = tf.keras.layers.Dropout(rate=float(dropout_rate))(hidden)

        return hidden


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Basic training arguments
    parser.add_argument("--batch-size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # Network architecture parameters
    parser.add_argument("--cnn", default="GA", type=str, help="Define architucture of the network as in mnist_cnn.py")
    # Regularization parameters
    parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
    parser.add_argument("--label-smoothing", default=0, type=float, help="Label smoothing.")
    # Optimizer parameters
    parser.add_argument("--optimizer", default='Adam', type=str, help="NN optimizer")
    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--learning-rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning-rate-final", default=None, type=float, help="Final learning rate.")
    # Transfer learning with resnet
    parser.add_argument("--resnet", default=None, type=str, help="Use resnet V2 model to use")
    parser.add_argument("--resnet-weights", default=None, type=str, help="use 'imagenet' to apply pretrained weights")
    #parser.add_argument("--num-ft", default=10, type=int, help="Number of top leayers to be fine tuned.")
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
    cifar = CIFAR10()

    args.train_size = cifar.train.size

    cifar_network = Network(args)
    cifar_network.train(cifar, args)
    cifar_network.test(cifar, args)
    cifar_network.save(args)


