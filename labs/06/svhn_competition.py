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
import sys

from roi_pooling import RoIPooling
from object_detection.anchor_generators.multiscale_grid_anchor_generator import MultiscaleGridAnchorGenerator



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


def fast_cnn_loss(gold, predicted):
    weights = tf.cast(tf.reduce_max(tf.abs(gold), axis=1) != 0., tf.float32)
    return tf.compat.v1.losses.huber_loss(gold, predicted) * weights

class Network:

    # i don't change those arguments per run

    LR_REDUCE_PRETRAINING = 0.01
    LR_REDUCE_PLATEAU = 0.2
    PLATEAU_PATIENCE = 10
    STOPPING_PATIENCE = 20

    MAX_SIZE = 224
    NUM_ANCHORS = 64
    PREDICTION_THRESHOLD = 0.4

    def __init__(self, args):
        tf.executing_eagerly()
        self.base_model = self.efficient_net(args)

        self.anchor_generator = MultiscaleGridAnchorGenerator(min_level=3, max_level=5, anchor_scale=4,
                                                              aspect_ratios=[0.5, 1., 2.], scales_per_octave=3, normalize_coordinates=False)

        self.anchors = self.anchor_generator.generate([(int(self.MAX_SIZE / 2**l), int(self.MAX_SIZE / 2**l)) for l in [3,4,5]]) #, im_height=self.MAX_SIZE, im_width=self.MAX_SIZE)] # [(1,1)]*3)]

        efficient_net = self.efficient_net(args)
        roi_inputs = [tf.keras.layers.Input([None,4], name='roi_input_{}'.format(l_idx)) for l_idx in [5,4,3]]

        faster_rcnn_outputs = self.FasterRCNN(roi_inputs, efficient_net.outputs)

        self._model = tf.keras.Model(inputs=[efficient_net.input] + roi_inputs, outputs=faster_rcnn_outputs)

        self._callbacks = []
        self._optimizer = self.get_optimizer(args)

        self.losses_names = ['sigmoid_focal_ce_0','sigmoid_focal_ce_1', 'sigmoid_focal_ce_2',
                             'huber_0', 'huber_1', 'huber_2']

        # self._model.compile(
        #     optimizer = self._optimizer,
        #     loss=[tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        #           tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        #           tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        #           fast_cnn_loss,
        #           fast_cnn_loss,
        #           fast_cnn_loss],
        #     metrics=[[tf.metrics.CategoricalAccuracy(name='accuracy_1')],
        #              [tf.metrics.CategoricalAccuracy(name='accuracy_2')],
        #              [tf.metrics.CategoricalAccuracy(name='accuracy_2')],
        #              [],[],[]])
        #
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

        # self._callbacks.append(tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1,
        #                                                         update_freq=100, profile_batch=0))
        # self._callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_iou',mode='max',
        #                                                         patience=self.STOPPING_PATIENCE,
        #                                                         restore_best_weights=True))

    def get_optimizer(self, args):
        learning_rate_final = args.learning_rate_final
        decay_steps = int(100 * args.epochs / args.batch_size)
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

    @tf.function
    def frcnn_loss(self, predictions, targets, weights):
        tot_loss = 0
        all_losses = []
        foreground = []
        for class_preds, class_tgts, wghts in zip(predictions[:3], targets[:3], weights):
            losses = tfa.losses.sigmoid_focal_crossentropy(class_preds, class_tgts)
            loss = tf.reduce_sum(losses * wghts)
            all_losses.append(loss)
            tot_loss += loss
            foreground.append(tf.cast(tf.reduce_any(class_tgts>0, axis=-1, keepdims=False), tf.float32))

        for frcnn_preds, frcnn_tgts, wghts in zip(predictions[3:], targets[3:], foreground):
            loss = tf.compat.v1.losses.huber_loss(frcnn_preds, frcnn_tgts, weights=tf.expand_dims(wghts, axis=-1))#reduction=tf.compat.v1.losses.Reduction.NONE) #weights=tf.broadcast_to(wghts, frnn_tgts.get_shape()))
            #loss = tf.reduce_sum(losses * wghts)
            tot_loss += loss
            all_losses.append(loss)

        return tot_loss, all_losses

    @tf.function
    def train_batch(self, inputs, targets, weights):
        with tf.GradientTape() as tape:
            predictions = self._model(inputs, training=True)
            loss, losses = self.frcnn_loss(predictions, targets, weights)
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return losses

    def train(self, svhn, args):

        train = svhn.train
        train = train.map(SVHN.parse)
        train = train.shuffle(500, seed=args.seed)
        train = train.map(lambda x: (x["image"], x['bboxes'], x["classes"]))
        train = train.map(self.preprocess_train)
        train = train.batch(args.batch_size).prefetch(args.threads)

        for epoch in range(args.epochs):
            for inputs, targets, weights in train.take(1):
                #self._model.train_on_batch(inputs, targets)
                losses = self.train_batch(inputs, targets, weights)

                # Generate the summaries each 10 steps
                if self._optimizer.iterations % 1 == 0:
                    print('BooooM!')
                    tf.summary.experimental.set_step(self._optimizer.iterations)
                    with self._writer.as_default():
                        for name, value in zip(self.losses_names, losses):
                            tf.summary.scalar("train/{}".format(name), value)

            self.evaluate(svhn.dev, 'validation', args)

    def evaluate(self, dataset, dataset_name, args):

        dataset = dataset.map(SVHN.parse)
        dataset = dataset.shuffle(500, seed=args.seed)
        dataset = dataset.map(lambda x: (x["image"], x['bboxes'], x["classes"]))
        dataset = dataset.map(self.preprocess_train)
        dataset = dataset.batch(args.batch_size).prefetch(args.threads)

        for inputs, targets, weights in dataset.take(10):
            predictions = self._model(inputs, training=False)
            _, losses = self.frcnn_loss(predictions, targets, weights)

        metrics = dict(zip(self.losses_names, losses))
        with self._writer.as_default():
            tf.summary.experimental.set_step(self._optimizer.iterations)
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        print("{} evaluation:".format(dataset_name))
        for name, value in metrics.items():
            print("\t{}: {}".format(name, value))

    def predict(self,dataset, dataset_name, args):

        dataset = dataset.map(SVHN.parse)
        dataset = dataset.map(lambda x: x["image"])
        dataset = dataset.map(self.preprocess_predict)
        dataset = dataset.batch(args.batch_size).prefetch(args.threads)

        with open(os.path.join(args.logdir, "{}_svhn_classification.txt".format(dataset_name)), "w", encoding="utf-8") as out_file:

            for input in dataset:
                preds = self._model(input, training=False)
                input_anchors = tf.concat([input[1], input[2], input[3]], axis=1)
                predicted_classes = tf.concat([preds[0], preds[1], preds[2]], axis=1)
                class_probs = tf.reduce_max(predicted_classes, axis=-1)
                predicted_classes = tf.argmax(predicted_classes, axis=-1)

                predicted_frcnn =tf.concat([preds[3], preds[4], preds[5]], axis=1)
                for exmpl_frcnn, exmpl_anchors, exmpl_class_prob, exmpl_class \
                    in zip(tf.unstack(predicted_frcnn), tf.unstack(input_anchors), tf.unstack(class_probs), tf.unstack(predicted_classes)):
                    print("ZuuuuuuM!")
                    exmpl_bboxes = tf.concat([bboxes_utils.bbox_from_fast_rcnn(anchor, frcnn) for anchor, frcnn
                                              in zip(tf.unstack(exmpl_anchors), tf.unstack(exmpl_frcnn))], axis=0)

                    indicies = tf.squeeze(tf.image.non_max_suppression(exmpl_bboxes, exmpl_class_prob, 300, 0.7, self.PREDICTION_THRESHOLD))
                    exmpl_bboxes = tf.gather(exmpl_bboxes, indicies).numpy()
                    exmpl_classes = tf.gather(exmpl_classes, indicies).numpy()

                    output = []
                    for bbox, pred_class in zip(exmpl_bboxes, exmpl_classes):
                        output.append(pred_class)
                        output.extend(bbox)

                    print(*output, file=out_file)

    def preprocess_train(self, image, bboxes, bbox_classes):
        tf.executing_eagerly()
        img_h = tf.shape(image)[0]

        image = tf.image.resize(image, [self.MAX_SIZE, self.MAX_SIZE])

        bboxes = tf.cast(bboxes, tf.float32) * tf.cast(self.MAX_SIZE / img_h, tf.float32)
        bbox_classes = bbox_classes

        layer_anchors = []
        layer_frcnns = []
        layer_classes = []
        layer_scores = []

        for l_anchors in self.anchors:

            l_anchors = l_anchors.get()
            anchor_classes, anchor_frcnns, scores= tf.numpy_function(bboxes_utils.bboxes_training,
                                                                    [l_anchors, bbox_classes, bboxes, 0.7],
                                                                    [tf.int32, tf.float32, tf.float32])
            # pos_indices = tf.random.shuffle(tf.where(anchor_classes > 0))[:self.NUM_ANCHORS]
            # neg_indices = tf.random.shuffle(tf.where(tf.logical_and(anchor_classes == 0, scores == 1)))[:self.NUM_ANCHORS]
            # sel_indices = tf.squeeze(tf.concat([pos_indices, neg_indices], 0))
            layer_anchors.append(l_anchors)
            layer_frcnns.append(anchor_frcnns)
            layer_classes.append(tf.one_hot(anchor_classes, depth=SVHN.LABELS +1))
            layer_scores.append(scores)

            layer_anchors.reverse()
            layer_frcnns.reverse()
            layer_classes.reverse()
            layer_scores.reverse()

        return tuple((tuple([image] + layer_anchors), tuple(layer_classes + layer_frcnns), tuple(layer_scores)))

    def preprocess_predict(self, image):
        img_h = tf.shape(image)[0]

        image = tf.image.resize(image, [self.MAX_SIZE, self.MAX_SIZE])

        # anchors_shape = anchors.shape
        layer_anchors = []

        for l_anchors in self.anchors:
            l_anchors = l_anchors.get()
            sel_indices = tf.squeeze(
                tf.where(
                    tf.reduce_all(
                        tf.logical_and(l_anchors >= 0, l_anchors <= self.MAX_SIZE), axis=-1)))
            layer_anchors.append(tf.gather(l_anchors, sel_indices))

        layer_anchors.reverse()

        return tuple([image] + layer_anchors)

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


    def FasterRCNN(self, rois, down_scaled_output,min_layer=3,max_layer=5):
        outputs_regressors = []
        outputs_classes = []

        # rois.reverse()
        down_scaled_output.reverse()

        filters = (1280, 112, 40)

        for idx, l_idx in enumerate(range(max_layer,min_layer-1,-1)):
            f_map = down_scaled_output[l_idx-1]
            layer_rois = rois[idx]
            up_filters = filters[idx]

            if idx == 0:
                x = f_map

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
                x = x + f_map #tf.keras.layers.Lambda(lambda y: tf.keras.backend.stop_gradient(y))(dso)

            roi_pooled = RoIPooling(3, 3)((x, layer_rois / self.MAX_SIZE))

            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name='flatten'))(roi_pooled)
            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='relu', name='fcA_{}'.format(idx)))(out)
            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5))(out)
            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='relu', name='fcB_{}'.format(idx)))(out)
            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5))(out)

            out_class = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(SVHN.LABELS +1, activation='softmax', kernel_initializer='zero'),
                                        name='dense_class_{}'.format(idx))(out)

            out_regr = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4 , activation='linear', kernel_initializer='zero'),
                                       name='dense_regress_{}'.format(idx))(out)

            outputs_classes.append(out_class)
            outputs_regressors.append(out_regr)

        return outputs_classes + outputs_regressors

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
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
    svhn = SVHN()

    network = Network(args)
    network.train(svhn, args)
    network.predict(svhn.dev, "dev", args)
    network.predict(svhn.test, "test", args)
