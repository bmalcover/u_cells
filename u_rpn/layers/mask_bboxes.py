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
                "image_size": self.__image_size,
            }
        )

        return config

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
            original_shape[0],  # channels
            original_shape[1],  # batch
            self.__image_size[0],
            self.__image_size[1],
        )

        masks = tf.reshape(masks, new_shape)
        masks = tf.transpose(masks, [0, 2, 3, 1])

        return masks
