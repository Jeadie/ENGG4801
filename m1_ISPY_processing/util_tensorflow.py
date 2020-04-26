from typing import List, Union

import tensorflow as tf


def bytes_feature(value: Union[bytes, str]) -> tf.train.Feature:
    """ Converts a byte or string data type into a bytes_list Tensorflow feature.
    """
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value: float) -> tf.train.Feature:
    """ Converts a float data type into a float_list Tensorflow feature.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value: int) -> tf.train.Feature:
    """ Converts an integer into a int64_list Tensorflow feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64List_feature(value: List[int]) -> tf.train.Feature:
    """ Converts an integer List into a int64_list Tensorflow feature.
        """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
