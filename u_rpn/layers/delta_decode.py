""" New keras layer to perform delta decoding of the RPN output.

The output of the RPN is a tensor of shape (batch_size, num_anchors, 4) where the last dimension
represents the delta coordinates of the anchor with respect to the ground truth. This delta must be
decoded to obtain the final coordinates of the anchors. In this module we implement the delta
decoding as a custom layer.

Written by: Miquel Miró Nicolau (UIB), 2022.
"""
from typing import Tuple

import tensorflow as tf
import tensorflow.keras.layers as keras_layer
from tensorflow.keras import backend as K

__all__ = ["DeltaDecoder"]


class DeltaDecoder(keras_layer.Layer):
    def __init__(
        self,
        anchors: tf.Tensor,
        output_size: int,
        size: Tuple[int, int],
        iou_threshold: float = 0.3,
        score_threshold: float = 0.9,
        *args: list,
        **kwargs: dict
    ):
        super().__init__(*args, **kwargs)

        self.__anchors = anchors
        self.__output_size = output_size
        self.__size = size
        self.__iou_threshold = iou_threshold
        self.__score_threshold = score_threshold

    def get_config(self) -> dict:
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable) containing the configuration of a
        layer. The same layer can be reinstantiated later (without its trained weights) from this
        configuration.

        The config of a layer does not include connectivity information, nor the layer class name.
        These are handled by `Network` (one layer of abstraction above).

        Note that `get_config()` does not guarantee to return a fresh copy of dict every time it is
        called. The callers should make a copy of the returned dict if they want to modify it.

        Returns:
            Python dictionary.
        """
        config = super().get_config().copy()
        config.update(
            {
                "anchors": self.__anchors.numpy().tolist(),
                "size": self.__size,
                "output_size": self.__output_size,
                "iou_threshold": self.__iou_threshold,
                "score_threshold": self.__score_threshold,
            }
        )

        return config

    def __decode_deltas(self, deltas: tf.Tensor) -> tf.Tensor:
        """Converts deltas to original coordinates system.

        Args:
            deltas: 4D tensor with all deltas.

        Returns:
            Tensor with the deltas converted to bounding boxes.
        """
        deltas_2 = K.exp(deltas[:, :, 2])
        deltas_3 = K.exp(deltas[:, :, 3])

        anchors_height = self.__anchors[:, :, 2] - self.__anchors[:, :, 0]
        anchors_width = self.__anchors[:, :, 3] - self.__anchors[:, :, 1]

        height = anchors_height * deltas_2
        width = anchors_width * deltas_3

        center_y = (deltas[:, :, 0] * anchors_height) + (
            self.__anchors[:, :, 0] + 0.5 * anchors_height
        )
        center_x = (deltas[:, :, 1] * anchors_width) + (
            self.__anchors[:, :, 1] + 0.5 * anchors_width
        )

        bboxes_y = center_y - 0.5 * height
        bboxes_x = center_x - 0.5 * width

        b_boxes = tf.stack(
            [
                bboxes_y / self.__size[1],
                bboxes_x / self.__size[0],
                (bboxes_y + height) / self.__size[1],
                (bboxes_x + width) / self.__size[0],
            ]
        )

        b_boxes = tf.transpose(b_boxes, perm=[1, 2, 0])

        return b_boxes

    @tf.autograph.experimental.do_not_convert
    @tf.function
    def call(self, deltas: tf.Tensor, objectness: tf.Tensor, *args, **kwargs):
        bboxes = self.__decode_deltas(deltas)
        bboxes = tf.expand_dims(bboxes, axis=-2)

        objectness = tf.expand_dims(objectness, axis=-1)

        bboxes, _, _, _ = tf.image.combined_non_max_suppression(
            boxes=bboxes[:, :, :],
            scores=objectness[:, :, 1],
            max_output_size_per_class=self.__output_size,
            max_total_size=self.__output_size,
            iou_threshold=self.__iou_threshold,
            score_threshold=self.__score_threshold,
        )

        return bboxes
