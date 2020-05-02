import sys
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

_logger = logging.getLogger("View Data")


def main(args: List[str]) -> None:
    """ Attempts to parse all the Images from a list TFRecords and displays
    each Series as an animated image of the 3D stack.

    Args:
        args: `view_data.py` followed by a list of TFRecord paths.
            e.g. ["view_data.py", "data_1.tfrecords", "data_2.tfrecords"]

    Return:
        None
    """
    for raw in iter(tf.data.TFRecordDataset(args[1:])):
        patient_details, image_stacks = get_images(raw)
        _logger.info(f"Patient details: {patient_details}")
        for name, image in image_stacks:
            visualise_image(name, image)


def visualise_image(name: str, image: np.array) -> None:
    """ Visualises a 3D stack of images in grayscale.

    Runs through each slice of the 3D image with a slight pause between to
    create an animation of the 3D stack.

    Args:
        name: The name of the 3D image to be displayed above the animation.
        image: A 3D image whereby the first index is the z dimension.
            i.e [i, :,:] is a 2D slice of the 3D image.

    Returns:
        None
    """
    for j in range(image.shape[0]):
        plt.imshow(image[j, ...], cmap="gray")
        plt.title(f"Name of series: {name}")
        plt.pause(0.005)
    plt.show()


def get_images(
    raw_example: tf.Tensor,
) -> Tuple[Dict[str, int], List[Tuple[str, np.array]]]:
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

            shape = list(example[k.replace("image", "shape")].values.numpy())
            # Remove unnecessary dimensions of length 1.
            shape = list(filter(lambda x: x != 1, shape))
            images.append((k, tf.reshape(example[k].values, shape)))

        except tf.errors.InvalidArgumentError as e:
            _logger.error(
                f"Could not pass image with key: {k}. Error: {e}. "
                f"Patient: {patient_details.get('patient_id', 'unknown')}"
            )
            continue

    return (patient_details, images)


if __name__ == "__main__":
    main(sys.argv)
