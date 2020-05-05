import os

import keras
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D
)

from base.model import BaseModel


class VGG16(BaseModel):
    def __init__(self, config: dict) -> None:
        """
        Create a model used to classify hand written images using the MNIST dataset
        :param config: global configuration
        """
        super().__init__(config)

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
        dense_units = 40# 4096

        input_shape = (256, 256, 1)
        img_input = Input(shape=input_shape)

        x = _vgg_block(img_input, 64, 2, activation, padding, "block1")
        # x = _vgg_block(x, 128, 2, activation, padding, "block2")
        # x = _vgg_block(x, 256, 3, activation, padding, "block3")
        # x = _vgg_block(x, 512, 3, activation, padding, "block4")


        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(dense_units,
                  activation=activation,
                  name='fc1',
                  kernel_regularizer=keras.regularizers.l2(0.01))(x)

        x = Dropout(p_dropout)(x)
        x = Dense(3, activation='softmax', name='predictions')(x)

        model = tf.keras.Model(inputs=img_input, outputs=x, name='vgg16')
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(lr=self.config.get("learning_rate", 0.002)),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        return model
        # return tf.keras.estimator.model_to_estimator(
        #     model, model_dir=self.config.get("model_dir", os.getcwd())
        # )


def _vgg_block(x: tf.Tensor, filters: int, convs: int, activation: str, padding: str, name: str, p_dropout: float = -1) -> tf.Tensor:
    for c in range(convs):
        print("x.shape", x.shape)
        x = Conv2D(filters, (3, 3),
                   activation=activation,
                   padding=padding,
                   name=f'{name}_conv{c+1}')(x)
        x = BatchNormalization()(x)

    if p_dropout != -1:
        x = Dropout(p_dropout)(x)

    return MaxPooling2D((2, 2), strides=(2, 2), name=f'{name}_pool')(x)