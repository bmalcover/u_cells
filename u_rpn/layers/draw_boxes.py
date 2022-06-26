""" Module containing a keras layer to draw bounding boxes.

Written by: Miquel Miró Nicolau (UIB), 2022.
"""
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as keras_layer

__all__ = ["DrawBoxes"]

tf.no_gradient("DrawBoundingBoxesV2")


class DrawBoxes(keras_layer.Layer):
    def __init__(
        self, image_size: Tuple[int, int, int, int], *args: list, **kwargs: dict
    ):
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

    def __number_of_slices(self) -> int:
        """Returns the number of slices in the image.

        The number of slices is the number of bounding boxes in the batch:
            number_of_slices = batch_size * number_of_bboxes

        Returns:
            int: Number of slices.
        """
        return self.__image_size[0] * self.__image_size[-1]

    def call(self, bboxes: tf.Tensor, *args: list, **kwargs: dict) -> tf.Tensor:
        """Method called for the forward pass of the layer.

        Args:
            bboxes: Tensor with the bounding boxes [batch, number_of_bboxes, [y1, x1, y2, x2]].
            *args: list of additional arguments.
            **kwargs: dictionary with additional arguments.

        Returns:
            Tensor with the bounding boxes drawn.
        """
        boxes = tf.reshape(bboxes, (-1, 1, 4))

        images = tf.image.draw_bounding_boxes(
            images=tf.zeros(
                (
                    self.__number_of_slices(),
                    self.__image_size[1],
                    self.__image_size[2],
                    1,
                )
            ),
            boxes=boxes,
            colors=np.array([[1.0, 0.0, 0.0]]),
        )

        images = tf.transpose(images, [3, 0, 1, 2])
        images = tf.reshape(
            images,
            [
                self.__image_size[0],
                self.__image_size[-1],
                self.__image_size[1],
                self.__image_size[2],
            ],
        )
        images = tf.transpose(images, [0, 2, 3, 1])

        return images
