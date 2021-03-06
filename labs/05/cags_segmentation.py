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

# this is needed to load Efficient NET keras model
def swish_activation(x):
    return (tf.keras.activations.sigmoid(x) * x)

tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish_activation)})

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

class Network:

    # i don't change those arguments per run

    LR_REDUCE_PRETRAINING = 0.01
    LR_REDUCE_PLATEAU = 0.2
    PLATEAU_PATIENCE = 10
    STOPPING_PATIENCE = 20

    def __init__(self, args):
        tf.executing_eagerly()

        self.base_model = self.efficient_net(args)

        output = self.up_scale(self.base_model.input, self.base_model.output, args)

        self._model = tf.keras.Model(inputs=self.base_model.input, outputs=output)

        self.cut_out = args.cut_out

        self._callbacks = []
        self._optimizer = self.get_optimizer(args)

        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
            metrics=[CAGSMaskIoU(name="iou")],
        )

        self._callbacks.append(tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1,
                                                                update_freq=100, profile_batch=0))
        self._callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_iou',mode='max',
                                                                patience=self.STOPPING_PATIENCE,
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
            self._callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_iou', mode='max',
                                                                        factor=self.LR_REDUCE_PLATEAU,
                                                                        patience=self.PLATEAU_PATIENCE,
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
        train = train.map(self.train_augment)
        train = train.batch(args.batch_size).prefetch(args.threads)

        dev = cags.dev
        dev = dev.map(CAGS.parse)
        dev = dev.map(lambda x: (x["image"], x["mask"]))
        dev = dev.batch(args.batch_size).prefetch(args.threads)

        if args.pretrain_epochs:

            self.model_history = self._model.fit(train,
                                                 epochs=args.pretrain_epochs,
                                                 validation_data=dev,
                                                 callbacks=self._callbacks)

            self.end_pretraining(args)

        self.model_history = self._model.fit(train,
                                             epochs=args.epochs,
                                             initial_epoch=args.pretrain_epochs,
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
                                      format(curr_date, max(self.model_history.history['val_iou'][-20:]))), include_optimizer=False)

    def train_augment(self, image, out_mask):
        if tf.random.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
            out_mask = tf.image.flip_left_right(out_mask)
        # image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 32, CAGS.W + 32)

        rnd_H_resize = tf.random.uniform([], minval=0, maxval=64, dtype=tf.int32)
        rnd_W_resize = tf.random.uniform([], minval=0, maxval=64, dtype=tf.int32)
        image = tf.image.resize(image, [CAGS.H + rnd_H_resize, CAGS.W + rnd_W_resize])
        out_mask = tf.image.resize(out_mask, [CAGS.H + rnd_H_resize, CAGS.W + rnd_W_resize])

        rnd_H_crop = tf.random.uniform([], minval=0, maxval=rnd_H_resize+1, dtype=tf.int32)
        rnd_W_crop = tf.random.uniform([], minval=0, maxval=rnd_W_resize+1, dtype=tf.int32)

        image = tf.image.crop_to_bounding_box(image, rnd_H_crop, rnd_W_crop, CAGS.H, CAGS.W)
        out_mask = tf.image.crop_to_bounding_box(out_mask, rnd_H_crop, rnd_W_crop, CAGS.H, CAGS.W)

        # global features, add noise
        if args.noise_std:
            image = image + tf.keras.backend.random_normal(shape=tf.shape(image), mean=0.0,
                                                           stddev=args.noise_std, dtype=tf.float32)

        #cut_out
        if self.cut_out:
            mask = np.ones((CAGS.H + 2*self.cut_out, CAGS.W + 2*self.cut_out, CAGS.C), np.float32)
            mean_pixels = np.zeros((CAGS.H + 2*self.cut_out, CAGS.W + 2*self.cut_out, CAGS.C), np.float32)
            #image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 2*self.cut_out, CAGS.W + 2*self.cut_out)
            rnd_H_cutout = np.random.randint(CAGS.H + self.cut_out)
            rnd_W_cutout = np.random.randint(CAGS.W + self.cut_out)

            mask[rnd_H_cutout:rnd_H_cutout + self.cut_out, rnd_W_cutout + self.cut_out, :] = 0
            mask = tf.constant(mask)
            mask = tf.image.resize_with_crop_or_pad(mask, CAGS.H, CAGS.W)

            mean_pixels[rnd_H_cutout:rnd_H_cutout + self.cut_out, rnd_W_cutout + self.cut_out, :] = np.array([0.15610054, 0.15610054, 0.15610054], np.float32)
            mean_pixels = tf.constant(mean_pixels)
            mean_pixels = tf.image.resize_with_crop_or_pad(mean_pixels, CAGS.H, CAGS.W)

            image = image * mask + mean_pixels
            out_mask = out_mask * mask

        return image, out_mask

    def efficient_net(self, args):

        efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False, drop_connect=args.drop_connect)

        for lidx, layer in enumerate(efficientnet_b0.layers):
            if ('conv' in layer.name or 'expand' in layer.name or 'reduce' in layer.name):
                layer.kernel_regularizer = tf.keras.regularizers.l2(args.l2)

            if ('drop' in layer.name or 'bg' in layer.name or 'gn' in layer.name):
                layer.trainable = True
            else:
                layer.trainable = False

        return efficientnet_b0

    def end_pretraining(self, args):
        num_layers = len(self.base_model.layers)
        for lidx, layer in enumerate(self.base_model.layers):
            if ('conv' in layer.name or 'expand' in layer.name or 'reduce' in layer.name) \
                    and lidx / num_layers >= args.freeze:
                layer.trainable = True
            elif ('drop' in layer.name or 'bg' in layer.name or 'gn' in layer.name):
                layer.trainable = True
            else:
                layer.trainable = False

        new_args = args

        new_args.learning_rate = args.learning_rate * self.LR_REDUCE_PRETRAINING
        self._optimizer = self.get_optimizer(args)
        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
            metrics=[CAGSMaskIoU(name="iou")],
        )

    def up_scale(self, input, down_scaled_output, args):
        for idx, dso in enumerate(down_scaled_output[1:] + [input]):
            up_filters = dso.shape[-1]

            if idx == 0:
                x = dso
            else:
                x = tf.keras.layers.Conv2DTranspose(filters=up_filters,
                                                    kernel_size=1,
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                                                    kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                                    name ='upscale_conv_' + str(idx))(x)
                x = tf.keras.layers.BatchNormalization(axis=-1, name='upscale_bn_' + str(idx))(x)
                x = tf.keras.layers.Activation(tf.nn.swish, name='upscale_activation_' + str(idx))(x)

                # Stopping horizontal gradient theoretically should be helpful, but it wasn't demonstrated in the experiments.
                x = x + dso #tf.keras.layers.Lambda(lambda y: tf.keras.backend.stop_gradient(y))(dso)

            x = tf.keras.layers.Conv2D(up_filters,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                kernel_initializer=CONV_KERNEL_INITIALIZER,
                                kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                name='horizontal_conv_' + str(idx))(x)

            x = tf.keras.layers.BatchNormalization(axis=-1, name='horizontal_bn_' + str(idx))(x)
            x = tf.keras.layers.Activation(tf.nn.swish, name='horizonatl_activation_' + str(idx))(x)

            x = tf.keras.layers.Conv2D(up_filters,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                kernel_initializer=CONV_KERNEL_INITIALIZER,
                                kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                name='horizontal_conv_' + str(idx) + 'b')(x)

            x = tf.keras.layers.BatchNormalization(axis=-1, name='horizontal_bn_' + str(idx) + 'b')(x)
            x = tf.keras.layers.Activation(tf.nn.swish, name='horizonatl_activation_' + str(idx) + 'b')(x)

        x = tf.keras.layers.Conv2D(1,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same',
                                   kernel_initializer=CONV_KERNEL_INITIALIZER,
                                   kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                   name='upscale_top_conv')(x)
        output = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name='upscale_top_activation')(x)
        return output

class NetworkEnsemble():

    def __init__(self, cags, args):
        self.ensemble_dir = args.ensemble_dir

        self.dev_masks = tf.constant([lab.numpy() for lab in cags.dev.map(CAGS.parse).map(lambda x: x["mask"])], tf.float32)
        # print(type(self.dev_labels[0]))
        dev_len = len(self.dev_masks)
        self.dev_res = np.zeros((dev_len, CAGS.H, CAGS.W, 1), np.float32)

        test_len = sum(1 for _ in cags.test.map(CAGS.parse))
        self.test_res = np.zeros((test_len, CAGS.H, CAGS.W, 1), np.float32)

    def predict(self, cags, args):

        dev = cags.dev
        dev = dev.map(CAGS.parse)
        dev = dev.map(lambda x: x["image"])
        dev = dev.batch(args.batch_size)

        test = cags.test
        test = test.map(CAGS.parse)
        test = test.map(lambda x: x["image"])
        test = test.batch(args.batch_size)

        num_model = 0
        for model_h5 in os.listdir(self.ensemble_dir):
            if model_h5.endswith('.h5'):
                num_model += 1
                print(model_h5)
                model = tf.keras.models.load_model(os.path.join(self.ensemble_dir,model_h5))
                self.dev_res += model.predict(dev)
                self.test_res += model.predict(test)

        if num_model:
            self.dev_res /= num_model
            self.test_res /= num_model

    def evaluate(self):
        mask_iou = CAGSMaskIoU(name='iou_ensemble')
        for dev_mask, dev_result in zip(self.dev_masks, self.dev_res):
            mask_iou.update_state(dev_mask, dev_result)
        dev_acc = mask_iou.result()
        with open(os.path.join(self.ensemble_dir, "dev_out"), "w", encoding="utf-8") as out_file:
            print("{:.4f}".format(dev_acc.numpy()), file=out_file)
        with open(os.path.join(args.ensemble_dir, "cags_segmentation.txt"), "w", encoding="utf-8") as out_file:
            for mask in self.test_res:
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
    parser.add_argument("--drop-connect", default=0.2, type=float, help="Drop connection probability in efficient net.")
    # Augmentation parameters
    parser.add_argument("--cut-out", default=0, type=int, help="Size of cut out window.")
    parser.add_argument("--noise-std", default=0., type=float, help="Std of gaussian noise added to image (<0.001)")
    # Optimizer parameters
    parser.add_argument("--optimizer", default='Adam', type=str, help="NN optimizer")
    parser.add_argument("--momentum", default=0., type=float, help="Momentum of gradient schedule.")
    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--learning-rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning-rate-final", default=None, type=float, help="Final learning rate.")
    # Transfer learning with resnet
    parser.add_argument("--pretrain-epochs", default=5, type=int, help="First epochs when ")
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

    if args.ensemble_dir:
        network_ensemble = NetworkEnsemble(cags, args)
        network_ensemble.predict(cags, args)
        network_ensemble.evaluate()

    else:
        cags_network = Network(args)
        cags_network.train(cags, args)
        cags_network.test(cags, args)
        cags_network.save(args, curr_date)

