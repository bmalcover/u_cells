# -*- coding: utf-8 -*-
""" Module containing all RPN classes and methods.

First proposed by Ren et al. a Region Proposal Network (RPN) takes an image (of any size) as input
and outputs a set of rectangular object proposals, each with an objectness score. This model is
processed with a fully-convolutional network.

"""
from typing import Tuple, Union
import enum

import tensorflow.keras.models as keras_model
import tensorflow.keras.layers as keras_layer
import tensorflow.keras.optimizers as keras_opt
import tensorflow as tf

from ..common import losses as own_losses
from ..model.base_model import BaseModel


class NeuralMode(enum.Enum):
    """ Mode for the Neural Network. """
    INFERENCE = 0
    TRAIN = 1


class RPN(BaseModel):
    """ RPN model based on Faster-RCNN architecture """

    def __init__(self, mode: NeuralMode, input_size: Tuple[int, int, int], feature_layer,
                 feature_depth: Union[int, float], mask_output, img_input, config):
        self.__config = config

        self.__feature_depth: int = feature_depth
        self.__feature_layer = feature_layer
        self.__mask_output = mask_output
        self.__img_input = img_input
        self.__mode = mode

        super().__init__(input_size)

    @staticmethod
    def __rpn_graph(feature_map, anchors_per_location, anchor_stride):
        """ Builds the computation graph of Region Proposal Network.

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
        anchor_score = keras_layer.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                                          activation='linear', name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = keras_layer.Lambda(
            lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 2]), name="rpn_class_reshape")(
            anchor_score)

        # Softmax on last dimension of BG/FG.
        rpn_probs = keras_layer.Activation(
            "softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        bbox = keras_layer.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                                  activation='linear', name='rpn_bbox_pred')(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = keras_layer.Lambda(lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 4]),
                                      name="rpn_bbox_reshape")(bbox)

        return [rpn_class_logits, rpn_probs, rpn_bbox], shared

    @staticmethod
    def __build_rpn_model(anchor_stride=1, anchors_per_location=3, depth=256,
                          input_feature_map=None):
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
        generate = input_feature_map is None
        if generate:
            input_feature_map = keras_layer.Input(shape=[None, None, depth],
                                                  name="input_rpn_feature_map")
        outputs, shared = RPN.__rpn_graph(input_feature_map, anchors_per_location, anchor_stride)

        if generate:
            return keras_model.Model([input_feature_map], outputs, name="rpn_model"), shared
        else:
            return outputs, shared

    def build_rpn(self, input_size, connection_layer=None):
        """ Builds the Region Proposal Network.

        Args:
            input_size: Input size of the feature map.
            connection_layer: The layer to connect the RPN to.
        Returns:

        """
        input_gt_masks = keras_layer.Input(
            shape=[input_size[0], input_size[1], None], name="input_gt_masks")

        rpn, rpn_conv = RPN.__build_rpn_model(self.__config.RPN_ANCHOR_STRIDE,
                                              len(self.__config.RPN_ANCHOR_RATIOS),
                                              self.__feature_depth, connection_layer)  # Conv5

        if connection_layer is None:
            if isinstance(self.__feature_layer, list):
                layer_outputs = []  # list of lists
                for feature_map in self.__feature_layer:
                    layer_outputs.append(rpn([feature_map]))
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
        else:
            rpn_output = rpn

        # RPN Output
        rpn_class_logits, rpn_class, rpn_bbox = rpn_output
        # rpn_conv = rpn.get_layer('rpn_conv_shared')

        return input_gt_masks, (rpn_class_logits, rpn_class, rpn_bbox), rpn_conv

    def build(self, input_gt_masks=None, rpn=None, mask_output=None, do_mask=True, mask_loss=None,
              *args, **kwargs):
        """ Builds the model.

        The RPN model building is done by the combination of the output of a backbone model. This
        backbone model had been passed previously in the constructor.

        Args:
            input_gt_masks: Input tensor of ground truth masks.
            rpn: Output of the RPN model.
            do_mask: Boolean if true, the model will build the mask branch.
            mask_output: Output of the mask model.
            mask_loss: Loss function of the mask output.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:

        """
        if mask_output is None:
            mask_output = self.__mask_output

        if input_gt_masks is None or rpn is None:
            input_gt_masks, rpn, _ = self.build_rpn(self.__img_input)

        rpn_class_logits, rpn_class, rpn_bbox = rpn

        if self.__mode is NeuralMode.TRAIN:
            # RPN GT
            input_rpn_match = keras_layer.Input(shape=[None, 1], name="input_rpn_match",
                                                dtype=tf.int32)
            input_rpn_bbox = keras_layer.Input(shape=[None, 4], name="input_rpn_bbox",
                                               dtype=tf.float32)
            input_gt_class_ids = keras_layer.Input(shape=[None], name="input_gt_class_ids",
                                                   dtype=tf.int32)

            if do_mask:
                if mask_loss is None:
                    mask_loss = keras_layer.Lambda(lambda x: own_losses.mrcnn_mask_loss_graph(*x),
                                                   name="img_out_loss")(
                        [input_gt_masks, input_gt_class_ids, mask_output])
                else:
                    mask_loss = keras_layer.Lambda(lambda x: mask_loss(*x), name="img_out_loss")(
                        [input_gt_masks, input_gt_class_ids, mask_output])

            # RPN Loss
            rpn_class_loss = keras_layer.Lambda(lambda x: own_losses.class_loss_graph(*x),
                                                name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = keras_layer.Lambda(lambda x: own_losses.bbox_loss_graph(*x),
                                               name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox, self.__config.BATCH_SIZE])

            # Input of the model
            inputs = [self.__img_input, input_rpn_match, input_rpn_bbox,
                      input_gt_class_ids]

            # Output of the model
            outputs = [rpn_class,
                       rpn_bbox,
                       rpn_class_loss,
                       rpn_bbox_loss]
            if do_mask:
                inputs.insert(1, input_gt_masks)
                outputs = [mask_output, mask_loss] + outputs

        else:
            # Create masks for detections
            inputs = [self.__img_input]
            outputs = [rpn_class, rpn_bbox]
            if do_mask:
                outputs = [mask_output] + outputs

        self._internal_model = keras_model.Model(inputs=inputs, outputs=outputs, name='rpn')

    def compile(self, do_mask=True, *args, **kwargs):
        """ Compiles the model.

        This function has two behaviors depending on the inclusion of the RPN. In the case of
        vanilla U-Net this function works as wrapper for the keras model compile method.

        Args:
            do_mask: Boolean if true, the model will compile the mask branch.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        loss_names = ["rpn_class_loss", "rpn_bbox_loss"]

        if do_mask:
            loss_names.append("img_out_loss")

        for name in loss_names:
            layer = self._internal_model.get_layer(name)
            loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * 1.0)
            self._internal_model.add_loss(loss)
            self._internal_model.add_metric(loss, name=name, aggregation='mean')

        self._internal_model.compile(*args, **kwargs,
                                     optimizer=keras_opt.Adam(lr=self.__config.LEARNING_RATE),
                                     loss=[None] * len(self._internal_model.outputs))

    def train(self, train_generator, val_generator, epochs: int, check_point_path: Union[str, None],
              callbacks=None, verbose=1, validation_steps=None, *args, **kwargs):
        """ Trains the model with the info passed as parameters.

        The keras model is trained with the information passed as parameters. The info is defined
        on Config class or instead passed as parameters.

        Args:
            train_generator: Generator for the training data.
            val_generator: Generator for the validation data.
            epochs (int): Number of epochs to train the model.
            check_point_path (str): Path to the file where the model will be saved.
            callbacks: List of callbacks to be used during the training.
            verbose (int): Verbosity mode.
            validation_steps (int | None): Number of steps of the validation data to process. If
                                            None, the value is obtained from the configuration file.

        Returns:
            History of the training.
        """
        if self.__mode is not NeuralMode.TRAIN:
            raise ValueError(
                f"Mode of the Neural network incorrect: instead of train the mode is {self.__mode}")

        steps_per_epoch = self.__config.STEPS_PER_EPOCH

        if validation_steps is None:
            validation_steps = self.__config.VALIDATION_STEPS

        return super().train(train_generator, val_generator, epochs, steps_per_epoch,
                             validation_steps,
                             check_point_path, callbacks, verbose)

    def predict(self, *args, **kwargs):
        """ Infer the value from the Model.

        When the model is the vanilla U-Net this method wrapper the original predict method of the
        keras model. In the case of U-Net + RPN (and if the raw parameters is set to False), the
        results are filtered depending on the value of the objectevness. This filter is a minimum
        threshold defined on the config object.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:

        """
        if self.__mode is NeuralMode.TRAIN:
            raise EnvironmentError("This method only can be called if the Mode is set to inference")

        pred_threshold = self.__config.PRED_THRESHOLD
        prediction = self._internal_model.predict(*args, **kwargs)

        if not self.__config.RAW_PREDICTION:
            masks, cls, bboxes = prediction

            bboxes_pred = []
            for batch_idx in range(bboxes.shape[0]):
                objectevness = cls[batch_idx, :, 1]
                bboxes_idx = bboxes[batch_idx, :, :]

                bboxes_idx = bboxes_idx[objectevness > pred_threshold]
                bboxes_pred.append(bboxes_idx)

            prediction = [masks, bboxes_pred]

        return prediction

    @staticmethod
    def features_2_rpn(features, depth: int):
        """ Prepares a list of features layer to be used on a RPN method as a pyramid feature layer

        Args:
            features (list): List of layers that represent a set o features of different size
            depth (int): Depth of the resulting layers.

        Returns:
            list: List of layers that represent a set o features of different size
        """
        rpn_features = []

        for feature_map in features:
            feature_map = keras_layer.Conv2D(depth, (1, 1))(feature_map)
            if rpn_features:
                feature_map = keras_layer.Add()(
                    [keras_layer.UpSampling2D(size=(2, 2))(rpn_features[-1]), feature_map])
            feature_map = keras_layer.Conv2D(depth, (3, 3), padding="same")(feature_map)
            rpn_features.append(feature_map)

        return rpn_features
