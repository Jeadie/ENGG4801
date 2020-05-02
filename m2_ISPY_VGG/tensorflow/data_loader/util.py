from typing import Dict, List, Tuple
import logging

import numpy as np
import tensorflow as tf

_logger = logging.getLogger("Data Util")


def int64List_feature(value: List[int]) -> tf.train.Feature:
    """ Converts an integer List into a int64_list Tensorflow feature.
        """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_images(raw_example: tf.Tensor) -> Tuple[Dict[str, int], List[Tuple[str, np.array]]]:
    """ Parses a single Patient proto from a TFRecord and retrieves specific
    patient data and all the images from the proto.

    Args:
        raw_example: A Tensor of type tf.string whereby the Tensor's value is
            the string proto of a tf.Example. i.e. One can create a tf.Example
            from the Tensor's value.

    Returns: Two elements in a Tuple:
        1. A mapping of the patient specific metadata. Currently, these fields
            are: `patient_id`, `pCR`, `RCB`.
        2. A list of 3D nd-array images for the patient. Each element consists
            of (as a Tuple) the name of the imaging Series (as found in the
            `raw_example`) and a 3D np.array of the image with shape as
            defined by its shape (again specified in the `raw_example`).
            e.g. ("timeX/MRI_RIGHT/image", np.array([[0,0,0...],..]]) )
    """
    string_example = tf.train.Example.FromString(raw_example.numpy())
    keys = list(
        filter(lambda x: "/image" in x, dict(string_example.features.feature).keys())
    )
    patient_features = {
        "patient_id": tf.io.FixedLenFeature((1), tf.int64),
        "pCR": tf.io.FixedLenFeature((1), tf.int64),
        "RCB": tf.io.FixedLenFeature((1), tf.int64),
    }
    patient_details = tf.io.parse_example(raw_example, patient_features)

    # Convert Features to ints.
    for k in patient_features:
        patient_details[k] = list(patient_details[k].numpy())[0]

    images = []
    # Parse_example image by image in case of error, can keep processing other images.
    for k in keys:
        try:
            features = {
                k: tf.io.VarLenFeature(tf.int64),
                k.replace("image", "shape"): tf.io.VarLenFeature(tf.int64),
            }
            example = tf.io.parse_example(raw_example, features)
            images.append((k, reconstruct_image(example[k], example[k.replace("image", "shape")])))

        except tf.errors.InvalidArgumentError as e:
            _logger.error(
                f"Could not pass image with key: {k}. Error: {e}. "
                f"Patient: {patient_details.get('patient_id', 'unknown')}"
            )
            continue

    return (patient_details, images)


def reconstruct_image(image: tf.Tensor, shape: tf.Tensor) -> tf.Tensor:
    """ Reconstructs a 3D image feature from a parsed_example that has undefined shape.

    Args:
        image: An eager tensor of a dimensionless image.
        shape: An eager tensor of the images shape. (i.e can be three of four dimensions)

    Returns:
         A 3D image feature with its original shape reapplied.
    """
    shape = list(filter(lambda x: x != 1, list(shape.values.numpy())))
    return tf.reshape(image.values, shape)
