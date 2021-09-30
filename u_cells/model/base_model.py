import warnings
from abc import ABC
from typing import Union, Tuple

import tensorflow as tf


class BaseModel(ABC):
    def __init__(self, input_size: Union[Tuple[int, int, int], Tuple[int, int]]):
        self._input_size = input_size

        self._internal_model = None
        self._history = None

    @property
    def history(self):
        return self._history

    @property
    def internal_model(self):
        return self._internal_model

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def compile(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, train_generator, val_generator, epochs: int, steps_per_epoch: int,
              validation_steps: int, check_point_path: Union[str, None], callbacks=None, verbose=1,
              *args, **kwargs):
        """ Trains the model with the info passed as parameters.

        The keras model is trained with the information passed as parameters.

        Args:
            train_generator:
            val_generator:
            epochs:
            steps_per_epoch:
            validation_steps:
            check_point_path:
            callbacks:
            verbose:

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
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(check_point_path, verbose=0,
                                                                save_weights_only=False,
                                                                save_best_only=True))

        if val_generator is not None:
            history = self._internal_model.fit(train_generator, validation_data=val_generator,
                                               epochs=epochs,
                                               validation_steps=validation_steps,
                                               callbacks=callbacks,
                                               steps_per_epoch=steps_per_epoch,
                                               verbose=verbose, *args, **kwargs)
        else:
            history = self._internal_model.fit(train_generator, epochs=epochs,
                                               callbacks=callbacks, verbose=verbose,
                                               steps_per_epoch=steps_per_epoch, *args,
                                               **kwargs)

        self._history = history

    def get_layer(self, *args, **kwargs):
        """ Wrapper of the Keras get_layer function.
        """
        return self._internal_model.get_layer(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """ Infer the value from the Model, wrapper method of the keras predict.

        """
        return self._internal_model.predict(*args, **kwargs)

    def summary(self):
        self._internal_model.summary()

    def load_weights(self, path: str):
        self._internal_model.load_weights(path)
