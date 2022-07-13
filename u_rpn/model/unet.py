""" Module containing all functions to build the U-Net model.

This module contains the set of functions that defines the original U-Net networks. This network was
proposed by Ronnenberger et al. and is based from an Encoder-Decoder architecture.

Written by: Miquel Miró Nicolau (UIB)
"""
from abc import ABC
from typing import Callable, List, Optional, Tuple, Union

import tensorflow as tf
import tensorflow.keras.layers as keras_layer
import tensorflow.keras.models as keras_model
from tensorflow.keras import optimizers as opt

from u_rpn import branches as mm_branches
from u_rpn import layers as mm_layers
from u_rpn.model.base_model import BaseModel

__all__ = ["EncoderUNet", "DecoderUNet", "UNet"]


class CropConcatBlock(keras_layer.Layer):
    """Block of Keras. Concatenates and crop multiple layers."""

    @tf.autograph.experimental.do_not_convert
    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[
            :,
            height_diff : (x2_shape[1] + height_diff),
            width_diff : (x2_shape[2] + width_diff),
            :,
        ]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x


class UNet(BaseModel):
    """Class to build the UNet and the U-RPN.

    The UNet is build upon the combination of two different modules, defined in this same script:
    the encoder and the decoder.
    """

    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[int, int]],
        out_channel: int,
        batch_normalization: bool,
        residual: bool = False,
    ):
        super().__init__(input_size)

        self.__batch_normalization: bool = batch_normalization
        self.__n_channels: int = out_channel
        self.__residual: bool = residual

    def build(
        self,
        n_filters,
        last_activation: Union[Callable, str],
        dilation_rate: int = 1,
        layer_depth: int = 5,
        kernel_size: Tuple[int, int] = (3, 3),
        pool_size: Tuple[int, int] = (2, 2),
    ):
        """Builds the graph and model for the U-Net.

        The U-Net, first introduced by Ronnenberger et al., is an encoder-decoder architecture.
        Build through the stack of 2D convolutional and up sampling 2D.

        Args:
            n_filters:
            last_activation:
            dilation_rate:
            layer_depth:
            kernel_size:
            pool_size:

        """
        self._layers = {}

        encoder = EncoderUNet(
            input_size=self._input_size,
            residual=self.__residual,
            batch_normalization=self.__batch_normalization,
        )
        input_image, embedded = encoder.build(
            n_filters=n_filters,
            pool_size=pool_size,
            layer_depth=layer_depth,
            kernel_size=kernel_size,
        )

        self._layers["encoder"] = encoder

        decoder = DecoderUNet(
            input_size=self._input_size,
            residual=self.__residual,
            n_channels=self.__n_channels,
            batch_normalization=self.__batch_normalization,
        )
        mask_out = decoder.build(
            n_filters=n_filters,
            last_activation=last_activation,
            encoder=encoder,
            dilation_rate=dilation_rate,
            kernel_size=kernel_size,
            embedded=embedded,
        )

        self._layers["decoder"] = decoder

        model = keras_model.Model(inputs=input_image, outputs=mask_out)

        self._internal_model = model

        return input_image, encoder, decoder

    def compile(
        self,
        loss_func: Union[str, Callable] = "categorical_crossentropy",
        learning_rate: Union[int, float] = 3e-5,
        *args: list,
        **kwargs: dict,
    ):
        """Compiles the model.

        This function has two behaviors depending on the inclusion of the RPN. In the case of
        vanilla U-Net this function works as wrapper for the keras.model compile method.

        Args:
            loss_func (str | Callable): Loss function to apply to the main output of the U-Net.
            learning_rate (Num). Learning rate of the training

        Returns:

        """
        loss_functions = {"img_out": loss_func}

        self._internal_model.compile(
            *args,
            **kwargs,
            optimizer=opt.Adam(lr=learning_rate),
            loss=loss_functions,
            metrics=["categorical_accuracy"],
        )


class EncoderUNet(BaseModel, ABC):
    """Class that represents the Encoder model of the U-Net.

    Methods:
        build(
            n_filters,
            layer_depth: int = 5,
            kernel_size: Tuple[int, int] = (3, 3),
            pool_size: Tuple[int, int] = (2, 2),
            training: Optional[bool] = None,
        ):
            Builds the Encoder of the model.
    """

    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[int, int]],
        residual: bool = False,
        batch_normalization: bool = True,
    ):
        """Construct the model with the attributes for the encoder.

        Args:
            input_size: Tuple of the input size of the model.
            residual: Boolean to indicate if the residual connections should be used.
            batch_normalization: Boolean to indicate if the batch normalization should be used.
        """
        super().__init__(input_size)

        self.__residual = residual
        self.__batch_normalization = batch_normalization
        self._layers = None

    def build(
        self,
        n_filters: int,
        layer_depth: int = 5,
        kernel_size: Tuple[int, int] = (3, 3),
        pool_size: Tuple[int, int] = (2, 2),
        training: Optional[bool] = None,
        coord_conv: Optional[dict] = None,
    ) -> Tuple[keras_layer.Layer, keras_layer.Layer]:
        """Builds the encoder of the U-Net.

        Method that builds the encoder of the U-Net. The encoder is a stack of convolutional blocks
        that reduce the size of the input image to an embedding vector.

        Args:
            n_filters: Number of filters to use in the convolutional layers.
            layer_depth: Number of layers to use in the encoder.
            kernel_size: Tuple of the size of the kernel to use in the convolutional layers.
            pool_size: Tuple of the size of the pooling to use in the convolutional layers.
            training: Boolean to indicate if the model is in training mode.
            coord_conv: Dictionary with the size of the coordinate convolutional layer.
        Returns:

        """
        # Define input batch shape
        input_image = keras_layer.Input(self._input_size, name="input_image")
        self._layers = {}

        conv_params = dict(
            filters=n_filters,
            kernel_size=kernel_size,
            activation="relu",
            residual=self.__residual,
            batch_normalization=self.__batch_normalization,
        )

        x = input_image

        for layer_idx in range(0, layer_depth):
            conv_params["filters"] = n_filters * (2**layer_idx)

            coord_conv_size = None
            if coord_conv is not None and layer_idx in coord_conv:
                coord_conv_size = coord_conv[layer_idx]

            conv_params["coord_conv"] = coord_conv_size

            x = mm_layers.ConvBlock(
                layer_idx, name=f"e_conv_block_{layer_idx}", **conv_params
            )(x, training=training)
            self._layers[layer_idx] = x

            x = keras_layer.MaxPooling2D(pool_size, name=f"e_max_pool_{layer_idx}")(x)

        return input_image, x


class DecoderUNet(BaseModel, ABC):
    """Decoder of the U-Net.

    This class is responsible for the decoder part of the U-Net. It is a wrapper for the keras.model
    In addition to the vanilla decoder it also includes the posibility to use more advanced
    architectures with the parameters options:

    Args:
        input_size: Size of the input image.
        residual: Whether to use residual connections in the decoder.
        n_channels: Number of channels of the output image.
        class_output_size: By default None, if not None the decoder will have an extra branch for
            the classification of the output channels between if there are object in the channel or
            only background.
        merge_branch: Boolean, if true adds an extra output to the decoder that merges the masks.
        mask_product: Boolean , if true multiplies the masks to the output.
    """

    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[int, int], None],
        n_channels: int = 1,
        residual: bool = False,
        class_output_size=None,
        merge_branch: bool = False,
        batch_normalization: bool = True,
        mask_product: bool = False,
    ):
        super().__init__(input_size)

        self.__residual = residual
        self.__n_channels = n_channels
        self._filters_mc = class_output_size
        self.__merge_branch = merge_branch
        self.__batch_normalization = batch_normalization
        self.__mask_product = mask_product

    def build(
        self,
        n_filters: int,
        last_activation: Union[Callable, str],
        encoder: EncoderUNet,
        embedded,
        concat: Optional[dict] = None,
        product: Optional[dict] = None,
        dilation_rate: int = 1,
        kernel_size: Tuple[int, int] = (3, 3),
        coord_conv=None,
        training=None,
    ):
        """Builds the decoder of the U-Net.

        The decoder of the U-Net is responsible for the reconstruction of the input image. This
        implementation has multiples parameter to enhance the abilities of the decoder. The main
        goal is to be able to use the original decoder for instance segmentation.

        Args:
            n_filters: Integer, number of filters of the decoder.
            last_activation: Activation function of the last layer of the decoder.
            encoder: Encoder of the U-Net.
            embedded: Emmbedded vector of the encoder.
            concat: Dictionary of extra layers to be concatenated to the decoder.
            product: Dictionary of extra layers to be added, by product, to the decoder.
            dilation_rate: Integer, dilation rate of the decoder.
            kernel_size: Tuple of integers, kernel size of the decoder.
            coord_conv: Boolean, if true uses coordinades convolutional instead of vanilla
                        convolutional.
            training: Boolean, if true the model is always in training mode.

        Returns:
        """
        conv_params = dict(
            filters=n_filters,
            kernel_size=kernel_size,
            activation="relu",
            residual=self.__residual,
            batch_normalization=self.__batch_normalization,
        )

        self._layers = {}
        layer = embedded
        for layer_idx in range(len(encoder) - 1, -1, -1):
            conv_params["filters"] = n_filters * (2**layer_idx)

            layer = mm_layers.UpConvBlock(
                layer_idx,
                filter_size=(2, 2),
                filters=n_filters * (2**layer_idx),
                activation="relu",
                name=f"d_up_conv_block{layer_idx}",
            )(layer)

            encoder_layer = encoder[layer_idx]

            for op_name, layers, operation in [
                ("concatenate", concat, keras_layer.Concatenate),
                ("product", product, keras_layer.Multiply),
            ]:
                if layers is not None and layer_idx in layers:
                    encoder_layer = operation(
                        axis=-1, name=f"d_{op_name}_extra_{layer_idx}"
                    )([encoder_layer, concat[layer_idx]])

            layer = keras_layer.Concatenate(axis=-1, name=f"d_concatenate_{layer_idx}")(
                [layer, encoder_layer]
            )

            coord_conv_size = None
            if coord_conv is not None and layer_idx in coord_conv:
                coord_conv_size = coord_conv[layer_idx]

            layer = mm_layers.ConvBlock(
                layer_idx,
                coord_conv=coord_conv_size,
                name=f"d_conv_block_{layer_idx}",
                **conv_params,
            )(layer, training=training)

            self._layers[layer_idx] = layer

        output = self._build_output_mask(dilation_rate, last_activation, layer, product)

        return tuple(output)

    def _build_output_mask(
        self,
        dilation_rate: int,
        last_activation: Union[Callable, str],
        layer: keras_layer.Layer,
        product: dict,
    ) -> List[keras_layer.Layer]:
        """Build output mask.

        Args:
            dilation_rate: Integer, dilation rate of the decoder.
            last_activation: Activation function of the last layer of the decoder.
            layer:
            product:

        Returns:
            List of output layers.
        """
        out_mask = keras_layer.Conv2D(
            self.__n_channels,
            (1, 1),
            activation=last_activation,
            padding="same",
            dilation_rate=dilation_rate,
            kernel_initializer="he_normal",
            name="img_out",
        )(layer)

        if self.__mask_product and -1 in product:
            out_mask = keras_layer.Multiply()([out_mask, product[-1]])

        output = [out_mask]

        if self._filters_mc is not None:  # Mask class
            mask_class = mm_branches.MaskClass(
                filters=self._filters_mc, channels=self.__n_channels
            )(layer)

            out_mask = keras_layer.Multiply(name="multiply_end")([out_mask, mask_class])
            output = [out_mask, mask_class]

        if self.__merge_branch:
            merge_branch = mm_branches.MergeMasks(
                filters=self.__n_channels,
                dilation_rate=dilation_rate,
                last_activation=last_activation,
            )(out_mask)

            output.append(merge_branch)

        return output
