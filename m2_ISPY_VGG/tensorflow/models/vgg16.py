from typing import List
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    InputLayer,
    MaxPooling2D
)
import absl

from base.model import BaseModel

_logger = absl.logging

class VGG16(BaseModel):
    def model(self) -> tf.Tensor:
        """
        Define your model metrics and architecture, the logic is dependent on the mode.
        :param features: A dictionary of potential inputs for your model
        :param labels: Input label set
        :param mode: Current training mode (train, test, predict)
        :return: An estimator spec used by the higher level API
        """
        activation = self.config.get("activation", "relu")
        padding = self.config.get("padding", "same")
        p_dropout = self.config.get("p_dropout", 0.25)
        dense_units = 4096
        _logger.info(
            f"Building model with parameters: \n"
            f"Activation: {activation}\n"
            f"Padding Scheme: {padding}\n"
            f"Dropout: {p_dropout}\n"
            f"Dense units: {dense_units}\n"
        )

        input_shape = (256, 256, 19)
        img_input = Input(shape=input_shape)
        img_input = tf.identity(img_input, name="input")

        x = _vgg_block(img_input, 64, 2, activation, padding, "block1")
        x = _vgg_block(x, 128, 2, activation, padding, "block2")
        x = _vgg_block(x, 256, 3, activation, padding, "block3")
        x = _vgg_block(x, 512, 3, activation, padding, "block4")


        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(dense_units,
                  activation=activation,
                  name='fc1',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

        x = Dropout(p_dropout)(x)
        x = Dense(3, activation='softmax', name='predictions')(x)
    
        return construct(img_input, x, self.config, name="vgg16")

class babyVGG16(BaseModel):
    def model(self) -> tf.Tensor:
        """
        Define your model metrics and architecture, the logic is dependent on the mode.
        :param features: A dictionary of potential inputs for your model
        :param labels: Input label set
        :param mode: Current training mode (train, test, predict)
        :return: An estimator spec used by the higher level API
        """
        activation = self.config.get("activation", "relu")
        padding = self.config.get("padding", "same")
        p_dropout = self.config.get("p_dropout", 0.25)
        dense_units = 20
        _logger.info(
            f"Building model with parameters: \n"
            f"Activation: {activation}\n"
            f"Padding Scheme: {padding}\n"
            f"Dropout: {p_dropout}\n"
            f"Dense units: {dense_units}\n"
        )
        if self.config["use_stack"]:
            input_shape = (256, 256, 14)
        else:
            input_shape = (256, 256, 1)

        model = tf.keras.Sequential()
        model.add(InputLayer(input_shape=input_shape))

        # img_input = Input(shape=input_shape)
        # x = tf.identity(img_input, name="input")

        _vgg_block(model, 64, 2, activation, padding, "block1")
        _vgg_block(model, 32, 2, activation, padding, "block2")
        _vgg_block(model, 16, 2, activation, padding, "block3")

        # Classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(dense_units,
                  activation=activation,
                  name='fc1',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        model.add(Dropout(p_dropout))
        model.add(Dense(3, activation='softmax', name='predictions'))
    
        return construct(model, input_shape, self.config, name="vgg16")

def construct(model, input_shape, configuration, name=""):
    # model = tf.keras.Model(inputs=image, outputs=output, name=name)
    model.build(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(lr=configuration.get("learning_rate", 0.002)),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
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
