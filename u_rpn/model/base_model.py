""" This module contains a base model class.

The base model class is used to define the basic structure of a model. Is used to build the RPN
model and the U-Net model.

Written by Miquel Miró (UIB), 2021
"""

import warnings
from abc import ABC
from typing import Tuple, Union

import tensorflow as tf


class BaseModel(ABC):
    """Base model class.

    The base model class is used to define the basic structure of a model. Is used to build the RPN
    model and the U-Net model.

    Params:
        input_size (Tuple[int, int] | Tuple[int, int, int]): The input size of the model.
    """

    def __init__(self, input_size: Union[Tuple[int, int, int], Tuple[int, int]]):
        self._input_size = input_size

        self._internal_model = None
        self._history = None
        self._layers = None

    @property
    def history(self):
        """Returns the history of the training.

        Returns:
            History: History of the training, containing metrics by epoch
        """
        return self._history

    @property
    def internal_model(self):
        """Returns the keras model used as backbone of the class.

        Returns:
            Model: Keras model used as backbone of the class.
        """
        return self._internal_model

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def compile(self, *args, **kwargs):
        raise NotImplementedError

    def train(
        self,
        train_generator,
        val_generator,
        epochs: int,
        steps_per_epoch: int,
        validation_steps: int,
        check_point_path: Union[str, None],
        callbacks=None,
        verbose=1,
        *args,
        **kwargs
    ):
        """Trains the model with the info passed as parameters.

        The keras model is trained with the information passed as parameters. This method wraps the
        training method of the keras model. At the end of the training, the history of the training
        is saved into the history attribute of the model.

        Args:
            train_generator: The training data generator.
            val_generator: The validation data generator.
            epochs: The number of epochs to train the model.
            steps_per_epoch: The number of steps per epoch.
            validation_steps: The number of steps per validation.
            check_point_path: Save the model after each epoch.
            callbacks: The callbacks to use during the training.
            verbose: If true, print the training information.

        Returns:
            History: History of the training, containing metrics by epoch
        """
        if self._internal_model is None:
            raise ValueError("Before training the model should be build and compiled")

        if self._history is not None:
            warnings.warn("Model already trained, starting new training")

        if callbacks is None:
            callbacks = []

        if check_point_path is not None:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    check_point_path,
                    verbose=0,
                    save_weights_only=True,
                    save_best_only=True,
                )
            )

        if val_generator is not None:
            history = self._internal_model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                validation_steps=validation_steps,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                verbose=verbose,
                *args,
                **kwargs
            )
        else:
            history = self._internal_model.fit(
                train_generator,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose,
                steps_per_epoch=steps_per_epoch,
                *args,
                **kwargs
            )

        self._history = history

    def get_layer(self, *args, **kwargs):
        """Wrapper of the Keras get_layer function."""
        return self._internal_model.get_layer(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Infer the value from the Model, wrapper method of the keras predict."""
        return self._internal_model.predict(*args, **kwargs)

    def summary(self):
        self._internal_model.summary()

    def load_weights(self, path: str):
        self._internal_model.load_weights(path, by_name=True)

    def __len__(self):
        length = len(self._layers.keys()) if self._layers is not None else 0

        return length

    def __getitem__(self, key):
        if key not in self._layers:
            raise KeyError

        return self._layers[key]

    @property
    def layers(self):
        return self._layers if self._layers is not None else {}
