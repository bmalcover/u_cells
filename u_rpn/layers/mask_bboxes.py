""" Layer that draws bounding boxes on top of the image as masks.

Writen by: Miquel Miró Nicolau (UIB), 2022
"""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers as keras_layers


class MaskBboxes(keras_layers.Layer):
    """Keras layer to draw bounding boxes as masks.

    Args:
        image_size: Two int tuple with the image size.
    """

    def __init__(self, image_size: Tuple[int, int], *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)

        self.__image_size = image_size

    def __draw_bbox(self, bbox) -> tf.Tensor:
        """Draws a bounding box as a mask.

        Args:
            bbox:

        Returns:
            tf.Tensor: Mask with the bounding box.
        """
        pos_y, pos_x, height, width = tf.unstack(bbox)

        zeros_to_left = tf.zeros((pos_x, self.__image_size[1]))

        zeros_and_bbox = tf.concat(
            [
                tf.zeros((width, pos_y)),
                tf.ones((width, height)),
                tf.zeros((width, self.__image_size[1] - (pos_y + height))),
            ],
            axis=1,
        )
        zeros_to_right = tf.zeros(
            (self.__image_size[0] - (pos_x + width), self.__image_size[1])
        )

        return tf.concat([zeros_to_left, zeros_and_bbox, zeros_to_right], axis=0)

    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Tensor with the bounding boxes.
        Returns:
            tf.Tensor: Mask with the bounding boxes.
        """
        original_shape = tf.shape(inputs)

        boxes = tf.reshape(inputs, (-1, 4))

        masks = tf.map_fn(self.__draw_bbox, boxes)
        new_shape = (
            original_shape[0],
            original_shape[1],
            self.__image_size[0],
            self.__image_size[1],
        )

        return tf.reshape(masks, new_shape)
