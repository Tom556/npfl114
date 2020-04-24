#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import bboxes_utils
import efficient_net
from svhn_dataset import SVHN
import svhn_eval
import importlib



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


class RoIPooling(tf.keras.layers.Layer):
    """Implementation of Jamie Sevilla, available at medium.com"""
    def __init__(self, pooled_height, pooled_width, **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        super(RoIPooling, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height,
                self.pooled_width, n_channels)

    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI Pooling to a single image and a single ROI
        """
        # Compute the region of interest
        feature_map_height = int(feature_map.shape[0])
        feature_map_width = int(feature_map.shape[1])

        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width * roi[1], 'int32')
        h_end = tf.cast(feature_map_height * roi[2], 'int32')
        w_end = tf.cast(feature_map_width * roi[3], 'int32')

        region = feature_map[h_start:h_end, w_start:w_end, :]
        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width = w_end - w_start
        h_step = tf.cast(region_height / pooled_height, 'int32')
        w_step = tf.cast(region_width / pooled_width, 'int32')

        areas = [[(
            i * h_step,
            j * w_step,
            (i + 1) * h_step if i + 1 < pooled_height else region_height,
            (j + 1) * w_step if j + 1 < pooled_width else region_width
        )
            for j in range(pooled_width)]
            for i in range(pooled_height)]

        # Take the maximum of each area and stack the result
        def pool_area(x):
            return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0, 1])

        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """

        def curried_pool_roi(roi):
            return RoIPooling._pool_roi(feature_map, roi, pooled_height, pooled_width)

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output
        """

        def curried_pool_rois(x):
            return RoIPooling._pool_rois(x[0], x[1],
                                              self.pooled_height,
                                              self.pooled_width)

        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)
        return pooled_areas


def get_all_anchors(stride, sizes, ratios, max_size):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors
        ratios (tuple[int]): the aspect ratios of anchors
        max_size (int): maximum size of input image
    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.
    """
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on 0, have sqrt areas equal to the specified sizes, and aspect ratios as given.
    anchors = []
    for sz in sizes:
        for ratio in ratios:
            w = np.sqrt(sz * sz / ratio)
            h = ratio * w
            anchors.append([-w, -h, w, h])
    cell_anchors = np.asarray(anchors) * 0.5

    field_size = int(np.ceil(max_size / stride))
    shifts = (np.arange(0, field_size) * stride).astype("float32")
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = cell_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype("float32")
    return field_of_anchors



class Network:

    # i don't change those arguments per run

    LR_REDUCE_PRETRAINING = 0.01
    LR_REDUCE_PLATEAU = 0.2
    PLATEAU_PATIENCE = 10
    STOPPING_PATIENCE = 20

    MAX_SIZE = 64
    SCALES = np.array([2., np.power(2.,1./3.), np.power(2.,2./3.)]) * 4.

    def __init__(self, args):
        #print(tf.executing_eagerly())
        self.base_model = self.efficient_net(args)

        self.anchors = np.array([get_all_anchors(3, self.SCALES * np.power(2, l), (0.5, 1., 2.), self.MAX_SIZE)
                                 for l in range(3, 6)])

        # output = self.fast_rnn(self.base_model.input, self.base_model.output, args)
        #
        # self._model = tf.keras.Model(inputs=self.base_model.input, outputs=output)
        #
        # self.cut_out = args.cut_out
        #
        # self._callbacks = []
        # self._optimizer = self.get_optimizer(args)
        #
        # self._model.compile(
        # )
        #
        # self._callbacks.append(tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1,
        #                                                         update_freq=100, profile_batch=0))
        # self._callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_iou',mode='max',
        #                                                         patience=self.STOPPING_PATIENCE,
        #                                                         restore_best_weights=True))

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

    def train(self, svhn, args):

        train = svhn.train
        train = train.map(SVHN.parse)
        train = train.map(lambda x: (x["image"], x['bboxes'], x["classes"]))
        train = train.shuffle(500, seed=args.seed)
        train = np.array([self.preprocess_train(imx, bbxs, clss) for imx, bbxs, clss in train])

        for train_example in train[:20]:
            print(train_example)
        # train = train.batch(args.batch_size).prefetch(args.threads)
        #
        # dev = cags.dev
        # dev = dev.map(CAGS.parse)
        # dev = dev.map(lambda x: (x["image"], x["mask"]))
        # dev = dev.batch(args.batch_size).prefetch(args.threads)
        #
        # for train_batch in train:
        #     pass


    def test(self, svhn, args):
        test = svhn.test
        test = test.map(SVHN.parse)
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


    def preprocess_train(self, image, bboxes, bbox_classes):
        tf.executing_eagerly()
        img_h = tf.shape(image)[0]

        image = tf.image.resize(image, [self.MAX_SIZE, self.MAX_SIZE])

        bboxes = np.ceil(bboxes.numpy() * self.MAX_SIZE / img_h)
        bbox_classes = bbox_classes.numpy()

        # anchors = np.array([get_all_anchors(3, cls.SCALES * np.power(2,l), (0.5, 1., 2.), cls.MAX_SIZE) for l in range(3,6)])
        anchor_classes, anchor_bboxes = bboxes_utils.bboxes_training(self.anchors.reshape((-1,4)), bbox_classes, bboxes, 0.7)
        anchor_bboxes = np.reshape(anchor_bboxes, self.anchors.shape)
        anchor_classes = np.reshape(anchor_classes, self.anchors.shape[:-1])

        considered = [np.where(layer_classes != -1) for layer_classes in anchor_classes]

        anchors = [layer_anchors[l_considered] for layer_anchors, l_considered in zip(self.anchors, considered)]
        anchor_bboxes = [layer_anchor_bboxes[l_considered] for layer_anchor_bboxes, l_considered in zip(anchor_bboxes, considered)]
        anchor_classes = [layer_classes[l_considered] for layer_classes, l_considered in zip(anchor_classes, considered)]

        return (image, anchors), (anchor_bboxes, anchor_classes)


    # def train_augment(self, image, out_mask):
    #     if tf.random.uniform([]) >= 0.5:
    #         image = tf.image.flip_left_right(image)
    #         out_mask = tf.image.flip_left_right(out_mask)
    #     # image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 32, CAGS.W + 32)
    #
    #     rnd_H_resize = tf.random.uniform([], minval=0, maxval=64, dtype=tf.int32)
    #     rnd_W_resize = tf.random.uniform([], minval=0, maxval=64, dtype=tf.int32)
    #     image = tf.image.resize(image, [CAGS.H + rnd_H_resize, CAGS.W + rnd_W_resize])
    #     out_mask = tf.image.resize(out_mask, [CAGS.H + rnd_H_resize, CAGS.W + rnd_W_resize])
    #
    #     rnd_H_crop = tf.random.uniform([], minval=0, maxval=rnd_H_resize+1, dtype=tf.int32)
    #     rnd_W_crop = tf.random.uniform([], minval=0, maxval=rnd_W_resize+1, dtype=tf.int32)
    #
    #     image = tf.image.crop_to_bounding_box(image, rnd_H_crop, rnd_W_crop, CAGS.H, CAGS.W)
    #     out_mask = tf.image.crop_to_bounding_box(out_mask, rnd_H_crop, rnd_W_crop, CAGS.H, CAGS.W)
    #
    #     # global features, add noise
    #     if args.noise_std:
    #         image = image + tf.keras.backend.random_normal(shape=tf.shape(image), mean=0.0,
    #                                                        stddev=args.noise_std, dtype=tf.float32)
    #
    #     #cut_out
    #     if self.cut_out:
    #         mask = np.ones((CAGS.H + 2*self.cut_out, CAGS.W + 2*self.cut_out, CAGS.C), np.float32)
    #         mean_pixels = np.zeros((CAGS.H + 2*self.cut_out, CAGS.W + 2*self.cut_out, CAGS.C), np.float32)
    #         #image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 2*self.cut_out, CAGS.W + 2*self.cut_out)
    #         rnd_H_cutout = np.random.randint(CAGS.H + self.cut_out)
    #         rnd_W_cutout = np.random.randint(CAGS.W + self.cut_out)
    #
    #         mask[rnd_H_cutout:rnd_H_cutout + self.cut_out, rnd_W_cutout + self.cut_out, :] = 0
    #         mask = tf.constant(mask)
    #         mask = tf.image.resize_with_crop_or_pad(mask, CAGS.H, CAGS.W)
    #
    #         mean_pixels[rnd_H_cutout:rnd_H_cutout + self.cut_out, rnd_W_cutout + self.cut_out, :] = np.array([0.15610054, 0.15610054, 0.15610054], np.float32)
    #         mean_pixels = tf.constant(mean_pixels)
    #         mean_pixels = tf.image.resize_with_crop_or_pad(mean_pixels, CAGS.H, CAGS.W)
    #
    #         image = image * mask + mean_pixels
    #         out_mask = out_mask * mask
    #
    #     return image, out_mask

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
        # self._model.compile(
        #     optimizer=self._optimizer,
        #     loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
        #     metrics=[CAGSMaskIoU(name="iou")],
        # )

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

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    # Regularization parameters
    parser.add_argument("--l2", default=0., type=float, help="L2 regularization.")
    parser.add_argument("--label-smoothing", default=0., type=float, help="Label smoothing.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout in top layer")
    parser.add_argument("--drop-connect", default=0.2, type=float, help="Drop connection probability in efficient net.")
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
    svhn = SVHN()

    network = Network(args)

    network.train(svhn, args)


    # Load the EfficientNet-B0 model
    # efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False, dynamic_shape=False)

    # # TODO: Create the model and train it
    # model = ...
    #
    # # Generate test set annotations, but in args.logdir to allow parallel execution.
    # with open(os.path.join(args.logdir, "svhn_classification.txt"), "w", encoding="utf-8") as out_file:
    #     # TODO: Predict the digits and their bounding boxes on the test set.
    #     for prediction in model.predict(...):
    #         # Assume that for the given prediction we get its
    #         # - `predicted_classes`: a 1D array with the predicted digits,
    #         # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
    #         # We can then generate the required output by
    #         output = []
    #         for label, bbox in zip(predicted_classes, predicted_bboxes):
    #             output.append(label)
    #             output.extend(bbox)
    #         print(*output, file=out_file)
