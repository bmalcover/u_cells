# -*- coding: utf-8 -*-
""" Module containing all RPN classes and methods.

First proposed by Ren et al. a Region Proposal Network (RPN) takes an image (of any size) as input
and outputs a set of rectangular object proposals, each with an objectness score. This model is
processed with a fully-convolutional network.

"""
from typing import Tuple, Union
import warnings
import enum

import tensorflow.keras.models as keras_model
import tensorflow.keras.layers as keras_layer
import tensorflow.keras.optimizers as keras_opt
import tensorflow as tf

from u_cells.common import losses as own_losses


class NeuralMode(enum.Enum):
    """ Mode for the Neural Network. """
    INFERENCE = 0
    TRAIN = 1


class RPN:

    def __init__(self, mode: NeuralMode, input_size: Tuple[int, int, int], feature_layer,
                 feature_depth: Union[int, float], mask_output, img_input, config):
        self.__config = config
        self.__input_size: Tuple[int, int, int] = input_size

        self.__feature_depth: int = feature_depth
        self.__feature_layer = feature_layer
        self.__mask_output = mask_output
        self.__img_input = img_input
        self.__mode = mode

        self.__internal_model = None
        self.__history = None

    @staticmethod
    def __rpn_graph(feature_map, anchors_per_location, anchor_stride):
        """Builds the computation graph of Region Proposal Network.

        Args:
            feature_map: backbone features [batch, height, width, depth]
            anchors_per_location: number of anchors per pixel in the feature map
            anchor_stride: Controls the density of anchors. Typically 1 (anchors for each pixel in
                           the feature map), or 2 (every other pixel).

        Returns:
            rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits
                              (before softmax)
            rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                      applied to anchors.
        """
        # Shared convolutional base of the RPN
        shared = keras_layer.Conv2D(512, (3, 3), padding='same', activation='relu',
                                    strides=anchor_stride, name='rpn_conv_shared')(feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = keras_layer.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                               activation='linear',
                               name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = keras_layer.Lambda(
            lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 2]))(x)

        # Softmax on last dimension of BG/FG.
        rpn_probs = keras_layer.Activation(
            "softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = keras_layer.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                               activation='linear', name='rpn_bbox_pred')(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = keras_layer.Lambda(lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 4]))(x)

        return [rpn_class_logits, rpn_probs, rpn_bbox]

    @staticmethod
    def __build_rpn_model(anchor_stride=1, anchors_per_location=3, depth=256):
        """ Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        Args:
            anchors_per_location: number of anchors per pixel in the feature map
            anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                           every pixel in the feature map), or 2 (every other pixel).
            depth: Depth of the backbone feature map.

        Returns:
            rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits
                             (before softmax)
            rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                      applied to anchors.
        """
        input_feature_map = keras_layer.Input(shape=[None, None, depth],
                                              name="input_rpn_feature_map")
        outputs = RPN.__rpn_graph(input_feature_map, anchors_per_location, anchor_stride)

        return keras_model.Model([input_feature_map], outputs, name="rpn_model")

    def build_model(self):
        """ Builds the model.

        The RPN model building is done by the combination of the output of a backbone model. This
        backbone model had been passed previously in the constructor.

        Returns:

        """
        input_gt_masks = keras_layer.Input(
            shape=[self.__input_size[0], self.__input_size[1], None], name="input_gt_masks")

        # We connect the U-Net to the RPN via the last CONV5 layer, the last layer of the decoder.
        rpn = RPN.__build_rpn_model(depth=self.__feature_depth)  # Conv5

        if type(self.__feature_layer) is list:
            layer_outputs = []  # list of lists
            for p in self.__feature_layer:
                layer_outputs.append(rpn([p]))
            # Concatenate layer outputs
            # Convert from list of lists of level outputs to list of lists
            # of outputs across levels.
            # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
            output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
            rpn_output = list(zip(*layer_outputs))
            rpn_output = [keras_layer.Concatenate(axis=1, name=n)(list(o))
                          for o, n in zip(rpn_output, output_names)]
        else:
            rpn_output = rpn([self.__feature_layer])

        # RPN Output
        rpn_class_logits, rpn_class, rpn_bbox = rpn_output

        if self.__mode is NeuralMode.TRAIN:
            # RPN GT
            input_rpn_match = keras_layer.Input(shape=[None, 1], name="input_rpn_match",
                                                dtype=tf.int32)
            input_rpn_bbox = keras_layer.Input(shape=[None, 4], name="input_rpn_bbox",
                                               dtype=tf.float32)
            input_gt_class_ids = keras_layer.Input(shape=[None], name="input_gt_class_ids",
                                                   dtype=tf.int32)

            # RPN Loss
            rpn_class_loss = keras_layer.Lambda(lambda x: own_losses.class_loss_graph(*x),
                                                name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = keras_layer.Lambda(lambda x: own_losses.bbox_loss_graph(*x),
                                               name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])

            mask_loss = keras_layer.Lambda(lambda x: own_losses.mrcnn_mask_loss_graph(*x),
                                           name="img_out_loss")(
                [input_gt_masks, input_gt_class_ids, self.__mask_output])

            # Input of the model
            inputs = [self.__img_input, input_gt_masks, input_rpn_match, input_rpn_bbox,
                      input_gt_class_ids]

            # Output of the model
            outputs = [self.__mask_output,
                       mask_loss,
                       rpn_class,
                       rpn_bbox,
                       rpn_class_loss,
                       rpn_bbox_loss]

        else:
            # Create masks for detections
            inputs = [self.__img_input]
            outputs = [self.__mask_output, rpn_class, rpn_bbox]

        self.__internal_model = keras_model.Model(inputs=inputs, outputs=outputs, name='rpn')

    def compile(self, *args, **kwargs):
        """ Compiles the model.

        This function has two behaviors depending on the inclusion of the RPN. In the case of
        vanilla U-Net this function works as wrapper for the keras.model compile method.

        Args:
            None
        Returns:
            None
        """
        loss_names = ["rpn_class_loss", "rpn_bbox_loss", "img_out_loss"]

        for name in loss_names:
            layer = self.__internal_model.get_layer(name)
            loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * 1.0)
            self.__internal_model.add_loss(loss)

        self.__internal_model.compile(*args, **kwargs,
                                      optimizer=keras_opt.Adam(lr=self.__config.LEARNING_RATE),
                                      loss=[None] * len(self.__internal_model.outputs))

    def train(self, train_generator, val_generator, epochs: int, check_point_path: Union[str, None],
              callbacks=None, verbose=1, *args, **kwargs):
        """ Trains the model with the info passed as parameters.

        The keras model is trained with the information passed as parameters. The info is defined
        on Config class or instead passed as parameters.

        Args:
            train_generator:
            val_generator:
            epochs (int):
            check_point_path (str):
            callbacks:
            verbose (int):

        Returns:

        """
        if self.__mode is not NeuralMode.TRAIN:
            raise ValueError(
                f"Mode of the Neural network incorrect: instead of train the mode is {self.__mode}")

        if self.__history is not None:
            warnings.warn("Model already trained, starting new training")

        steps_per_epoch = self.__config.STEPS_PER_EPOCH
        validation_steps = self.__config.VALIDATION_STEPS

        if callbacks is None:
            callbacks = []

        if check_point_path is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(check_point_path, verbose=0,
                                                                save_weights_only=False,
                                                                save_best_only=True))

        if val_generator is not None:
            history = self.__internal_model.fit(train_generator, validation_data=val_generator,
                                                epochs=epochs,
                                                validation_steps=validation_steps,
                                                callbacks=callbacks,
                                                steps_per_epoch=steps_per_epoch,
                                                verbose=verbose, *args, **kwargs)
        else:
            history = self.__internal_model.fit(train_generator, epochs=epochs,
                                                callbacks=callbacks, verbose=verbose,
                                                steps_per_epoch=steps_per_epoch, *args,
                                                **kwargs)

        self.__history = history

    def predict(self, *args, **kwargs):
        """ Infer the value from the Model.

        When the model is the vanilla U-Net this method wrapper the original predict method of the
        keras model. In the case of U-Net + RPN (and if the raw parameters is set to False), the
        results are filtered depending on the value of the objectevness. This filter is a minimum
        threshold defined on the config object.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        if self.__mode is NeuralMode.TRAIN:
            raise EnvironmentError("This method only can be called if the Mode is set to inference")

        pred_threshold = self.__config.PRED_THRESHOLD
        masks, cls, bboxes = self.__internal_model.predict(*args, **kwargs)

        bboxes_pred = []
        for batch_idx in range(bboxes.shape[0]):
            objectevness = cls[batch_idx, :, 0]
            bboxes_idx = bboxes[batch_idx, :, :]

            bboxes_idx = bboxes_idx[objectevness > pred_threshold]
            bboxes_pred.append(bboxes_idx)

        prediction = [masks, bboxes_pred]
        return prediction

    def load_weights(self, path: str):
        self.__internal_model.load_weights(path)
