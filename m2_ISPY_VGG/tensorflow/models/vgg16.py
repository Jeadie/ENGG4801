from typing import List
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    InputLayer,
    MaxPooling2D,
    MaxPooling3D
)
import absl

from base.model import BaseModel

_logger = absl.logging

class VGG16(BaseModel):
    def model(self) -> tf.Tensor:
        """
        """
        activation = self.config["activation"]
        padding = self.config["padding"]
        p_dropout = self.config["p_dropout"]
        dense_units = self.config["dense_units"]
        _logger.info(
            f"Building model with parameters: \n"
            f"Activation: {activation}\n"
            f"Padding Scheme: {padding}\n"
            f"Dropout: {p_dropout}\n"
            f"Dense units: {dense_units}\n"
        )
        if not self.config["use_stack"]:
            input_shape = (256, 256, 50)
        else:
            input_shape = (256, 256, 1)

        model = tf.keras.Sequential()
        model.add(InputLayer(input_shape=input_shape))

        self.create_convs(model, activation, padding)

        # Classification block
        model.add(Flatten(data_format="channels_first", name='flatten'))
        model.add(Dense(dense_units,
                        activation=activation,
                        name='fc1',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        model.add(Dropout(p_dropout))
        model.add(Dense(3, activation='softmax', name='predictions'))

        return construct(model, input_shape, self.config, name="vgg16")

    def create_convs(self, model, activation, padding):
        _vgg_block(model, 64, 2, activation, padding, "block1")
        _vgg_block(model, 128, 2, activation, padding, "block2")
        _vgg_block(model, 256, 3, activation, padding, "block3")
        _vgg_block(model, 512, 3, activation, padding, "block4")


class babyVGG16(VGG16):
    def create_convs(self,model, activation, padding):
        _vgg_block(model, 64, 2, activation, padding, "block1")
        _vgg_block(model, 32, 2, activation, padding, "block2")
        _vgg_block(model, 16, 2, activation, padding, "block3")


def construct(model, input_shape, configuration, name=""):
    model.build(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(lr=configuration.get("learning_rate", 0.002)),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model

def _vgg_block(model: tf.keras.Sequential, filters: int, convs: int, activation: str, padding: str, name: str, p_dropout: float = -1):
    for c in range(convs):
        model.add(Conv2D(filters, (3, 3),
                   activation=activation,
                   padding=padding,
                   name=f'{name}_conv{c+1}'))
        model.add(BatchNormalization())

    if p_dropout != -1:
        model.add(Dropout(p_dropout))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name=f'{name}_pool'))
